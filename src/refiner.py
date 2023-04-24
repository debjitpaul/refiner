## This code is modified version of https://github.com/ypapanik/t5-for-code-generation 
import logging
from tokenizers import Token
import torch
from torch import nn
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import wandb
import torch.nn.functional as F

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup, T5ForConditionalGeneration, T5Tokenizer, T5Config
)

from src.data_processing.processor import load_and_cache_examples
from src.data_processing.utils import get_encoded_code_tokens
from src.eval.conala_eval import calculate_bleu_from_lists
from accelerate import Accelerator
accelerator = Accelerator()

logger = logging.getLogger(__name__)

import torch.distributed as dist

class REFINER:
    def __init__(self,
                 max_seq_length,
                 output_model_dir,
                 output_critique_model, 
                 pretrained_model_name_or_path,
                 number_turn,
                 exploration_number,
                 threads=4,
                 cache_dir='data/pretrained/',
                 do_lower_case=True,
                 local_rank=-1,
                 fp16=False,
                 fp16_opt_level='01',
                 ):
        self.max_seq_length = max_seq_length
        self.output_model_dir = output_model_dir
        self.output_critique_model = output_critique_model
        self.number_turn=number_turn,
        self.exploration_number=exploration_number,

        self.logger = logging.getLogger(__name__)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.cache_dir = cache_dir
        self.threads = threads
        # Setup logging
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)
        
        self.local_rank = local_rank
        self.fp16 = fp16

        # Setup CUDA, GPU & distributed training
        if local_rank == -1:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.n_gpu = torch.cuda.device_count()
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(local_rank)
            self.device = torch.device("cuda", local_rank)
            torch.distributed.init_process_group(backend="nccl")
            self.n_gpu = 1

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if local_rank in [-1, 0] else logging.WARN,
        )
        logger.warning(
            "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
            local_rank,
            self.device,
            self.n_gpu,
            bool(local_rank != -1),
            self.fp16,
        )

        if local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        self.config = T5Config.from_pretrained(pretrained_model_name_or_path=self.pretrained_model_name_or_path,
                                                 cache_dir=self.cache_dir)
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path=self.pretrained_model_name_or_path,
                                                     do_lower_case=do_lower_case,
                                                     cache_dir=self.cache_dir)
        
    def train(self, training_file,
              dev_file,
              per_gpu_train_batch_size,
              gradient_accumulation_steps,
              num_train_epochs,
              learning_rate,
              number_turn,
              exploration_number,
              weight_decay=0.0,
              warmup_steps=0,
              adam_epsilon=1e-8,
              fp16_opt_level='O1',
              max_grad_norm=1.0,
              optimizer_algorithm='adam', noisy_file=None):

        """ Train the model """
        train_batch_size = per_gpu_train_batch_size * max(1, self.n_gpu)
        train_dataset, train_labels = load_and_cache_examples(data_file=training_file, local_rank=self.local_rank,
                                                   max_seq_length=self.max_seq_length, tokenizer=self.tokenizer,
                                                   evaluate=False)

        val_dataset, val_labels = load_and_cache_examples(data_file=dev_file, local_rank=self.local_rank,
                                                          max_seq_length=self.max_seq_length, tokenizer=self.tokenizer,
                                                          evaluate=True)
        
        train_sampler = RandomSampler(train_dataset) if self.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)
        t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs
        
        # print(max_memory, free_in_GB, n_gpus)
        model = T5ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path="google/flan-t5-base",
            from_tf=bool(".ckpt" in self.pretrained_model_name_or_path),
            config=self.config,
            cache_dir=self.cache_dir,
        )
        
        model.to(self.device)

        per_gpu_eval_batch_size = 8 

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        
        policy_optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]

        if optimizer_algorithm == 'adam':
            policy_optimizer = Adam(policy_optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        else:
            policy_optimizer = AdamW(policy_optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
            
        policy_scheduler = get_linear_schedule_with_warmup(
            policy_optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )

        if self.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True
            )
            
        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", per_gpu_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            train_batch_size
            * gradient_accumulation_steps
            * (torch.distributed.get_world_size() if self.local_rank != -1 else 1),
        )
        logger.info("  Gradient Accumulation steps = %d", gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        tr_loss, logging_loss = 0.0, 0.0
        val_bleu = 0
        val_exact = 0
        factor = 1000
        early_stopping_counter = 0
        alpha = 0.7

        model.zero_grad()
        # critique_model.zero_grad()
        train_iterator = trange(epochs_trained, int(num_train_epochs), desc="Epoch", disable=self.local_rank not in [-1, 0])
        print(len(train_dataset))
        save_steps = len(train_dataset) // (per_gpu_train_batch_size * gradient_accumulation_steps* self.n_gpu)
        sep_tokens = self.tokenizer.batch_encode_plus(" <sep> ", padding=True, return_tensors="pt").input_ids.to(self.device) 
        cache_memory = [] 
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=self.local_rank not in [-1, 0])
            for step, batch in enumerate(epoch_iterator):
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                model.train()

                batch = tuple(t.to(self.device) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}#, 'label_mask': batch[3]}
                _input = batch[0] #input to the equation generation model
                _attention = batch[1] # attention 
                _labels = batch[2]
                turn_loss = 0
                loss = 0
                reward = 0
                count = 0

                for turn in range(1, int(number_turn)+1):
                    outputs = model(**inputs) # equation generation model 
                    turn_loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
                    
                    count+=1
                    # generate equation 
                    token_ids, tokens  = self.get_sample(input_ids=_input, 
                            attention_mask=_attention,
                            labels = _labels,
                            model=model,
                            max_generated_tokens=50,
                            exploration_number=exploration_number) 
                    labels = self._tokenids2tokens(batch[2])
                    
                    c = 0 
                    regret = 0
                    h_best = ["No <hint>"]
                    hints = []
                    regret_batch = 0 
                    best_tokens = []
                    r = 0
                    t = ""
                    explore_loss = 0
                    best_loss = 0
                    
                    for i in range(len(_input)): # number of sequences = #batch_size * gradient_accumulation_steps
                        explore_loss = 0
                        r = 0 
                        regret = 100000
                        for j in range(exploration_number):
                            
                            #### auto critic ###
                            #r , h_tok = self._critique_gen(critique_model, _input[i], tokens[c], token_ids[c], labels[i]) 
                            
                            
                            ##### oracle critic ####
                            r , h_tok = self._critique_function(tokens[c], labels[i])
                            if r <= regret:
                                h_best = h_tok
                                regret = r
                                t = tokens[c]
                            else:
                                h_best = h_best
                            c+=1
                        best_loss += explore_loss
                        best_tokens.extend([" Previous Answer: "+t])
                        regret_batch += regret
                        hints.extend(h_best)

                    regret_batch = regret_batch/(len(_input))
                    try:
                        bleu, exact = calculate_bleu_from_lists(gold_texts=labels,
                                                          predicted_texts=best_tokens)
                    except ZeroDivisionError:
                        bleu = 0
                        exact = 0
                     
                    hint_ids = self.tokenizer.batch_encode_plus(hints, padding=True, return_tensors="pt").input_ids.to(self.device)
                    token_ids = self.tokenizer.batch_encode_plus(best_tokens, padding=True, return_tensors="pt").input_ids.to(self.device) 
                    _input = torch.cat((batch[0], token_ids, hint_ids), 1)
                    _input = self.batch_move_zeros(_input).to(self.device)
                    _attention = _input.clone()
                    _attention[_input!=0] = 1
                    _attention.to(self.device)
                    
                    if turn == 1:
                        prev_token_ids = token_ids
                        prev_hint_ids = hint_ids

                    inputs = {"input_ids": _input, "attention_mask": _attention, "labels": batch[2]}
                    labels = self._tokenids2tokens(batch[2])
                    #if turn==1: 
                    loss += turn_loss
                    
                actor_loss = loss
                if self.n_gpu > 1:
                    actor_loss = actor_loss.mean()  # mean() to average on multi-gpu parallel training
                
                if gradient_accumulation_steps > 1:
                    actor_loss = actor_loss / gradient_accumulation_steps
                
                total_loss = actor_loss 
                
                if self.fp16:
                    with amp.scale_loss(actor_loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                elif loss is not None:
                    actor_loss.backward() 
                
                tr_loss += actor_loss.item()
                if (step + 1) % gradient_accumulation_steps == 0:
                    
                    if self.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(policy_optimizer), max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                    policy_optimizer.step()
                    policy_scheduler.step()  # Update learning rate schedule
                    
                    model.zero_grad()
                    
                    global_step += 1

                    if self.local_rank in [-1, 0] and save_steps > 0 and global_step % save_steps == 0:
                        # Log metrics
                        if self.local_rank == -1:  # Only evaluate when single GPU otherwise metrics may not avg well
                            wandb.log({"labels": labels,  "tokens": tokens, "hints": hints})
                            # Create tabular data, method 2
                            table = wandb.Table(columns=[" Correct Equation ", " Predicted Equation ",  " Hints "])
                            table.add_data(labels, best_tokens, hints)
                            wandb.log({"Output Table": table})
                            preds = self._predict(eval_dataset=train_dataset,
                                                  per_gpu_eval_batch_size=per_gpu_eval_batch_size,
                                                  model=model,
                                                  max_generated_tokens=56) 
                            labels = [' '.join(get_encoded_code_tokens(label)) for label in train_labels]
                            bleu, exact = calculate_bleu_from_lists(gold_texts=labels,
                                                           predicted_texts=preds)
                            wandb.log({'train_exact_match': exact, 'train_bleu': bleu})
                            print(exact, bleu)
                            wandb.log({'train_exact_match': exact, 'reward': reward})
                            preds = self._predict(eval_dataset=val_dataset,
                                                  per_gpu_eval_batch_size=per_gpu_eval_batch_size,
                                                  model=model,
                                                  max_generated_tokens=56)
                            labels = [' '.join(get_encoded_code_tokens(label)) for label in val_labels]
                        
                            bleu, exact = calculate_bleu_from_lists(gold_texts=labels,
                                                           predicted_texts=preds)
                            
                            print(exact, bleu)
                            wandb.log({'val_exact_match': exact, 'val bleu': bleu})
                            wandb.log({'training_losses': turn_loss})
                            if exact > val_exact:
                                model_to_save = (
                                    model.module if hasattr(model, "module") else model
                                )  # Take care of distributed/parallel training
                                model_to_save.save_pretrained(self.output_model_dir)
                                print('Exact match on dev set improved:', exact, ' over ', val_exact, 'saving model to disk.')
                                val_exact = exact
                                early_stopping_counter = 0
                            else:
                                print('Exact match on dev set did not improve:', val_exact)
                                early_stopping_counter +=1 
                            
                            if early_stopping_counter==20:
                                return global_step, tr_loss / global_step
 
        return global_step, tr_loss / global_step

    def predict(self,
                test_file,
                per_gpu_eval_batch_size, max_generated_tokens):
        eval_dataset, _ = load_and_cache_examples(test_file, local_rank=self.local_rank,
                                                  max_seq_length=self.max_seq_length, tokenizer=self.tokenizer,
                                                  evaluate=True)
        model = T5ForConditionalGeneration.from_pretrained(self.output_model_dir)
        
        return self._predict(eval_dataset=eval_dataset,
                             per_gpu_eval_batch_size=per_gpu_eval_batch_size,
                             model=model,
                             max_generated_tokens=max_generated_tokens)

    def generate_hint(self, critique_model, input_ids, attention_mask):
        
        critique_model.to(self.device)
        # multi-gpu eval
        if self.n_gpu > 1 and not isinstance(critique_model, torch.nn.DataParallel):
            critique_model = torch.nn.DataParallel(critique_model)
        
        # Eval!
        preds = []
        critique_model.eval()
        max_generated_tokens = 50
        with torch.no_grad():
            if self.n_gpu > 1:
                outs = critique_model.module.generate(input_ids=input_ids.to(self.device),
                                            attention_mask=attention_mask.to(self.device),
                                            max_length = 50,
                                            return_dict_in_generate=True,
                                            do_sample = False,
                                            output_scores = True,
                                            output_hidden_states=True,
                                            num_return_sequences = 1)
            else:
                outs = critique_model.generate(input_ids=input_ids.to(self.device),
                                        attention_mask = attention_mask.to(self.device),
                                        max_length = max_generated_tokens,
                                        return_dict_in_generate=True,
                                        do_sample = False,
                                        output_scores = True,
                                        num_return_sequences = 1)
            
            dec = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs.sequences]
            
        return outs.sequences, dec
    
    
    
    def _critique_function_batch(self, generated_explanation, gold_explanation):
        '''
        ------------------------
        Parameter: 
        generated explantion: 
        gold explanation: 
        ------------------------
        Output: 
        Hints
        '''

        hints = []
        hints_ids = []
        regret = 0  

        for i in range(len(generated_explanation)):
            hint = [] 
            if gold_explanation[i] == generated_explanation[i]:
                hint = "No"
                regret = 0
            else:
                list_eq1 = gold_explanation[i].split(' ')
                list_eq2 = generated_explanation[i].split(' ') 
                if len(list_eq2)>len(list_eq1):
                    difference_position = [pos for pos in range(len(list_eq1)) if list_eq2[pos] != list_eq1[pos]]
                    hint, regret = self.gen_hint(list_eq2, difference_position, regret)
                else: 
                    difference_position = [pos for pos in range(len(list_eq2)) if list_eq2[pos] != list_eq1[pos]]
                    hint, regret = self.gen_hint(list_eq1, difference_position, regret)
                if gold_explanation.count("#")>generated_explanation.count("#"): 
                        hint = hint + "Add an operation. " 
                elif generated_explanation.count("#")>gold_explanation.count("#"): 
                        hint = hint + "Remove an operation. "
            hint = [hint+ " <sep> "]
            hints.extend(hint)
        
        hints_ids = self.tokenizer.batch_encode_plus(hints, padding=True, return_tensors="pt").input_ids
        
        return hints_ids.to(self.device), hints
    
    def _critique_function(self, generated_explanation, gold_explanation):
        '''
        ------------------------
        Parameter: 
        generated explantion: 
        gold explanation: 
        ------------------------
        Output: 
        Hints
        '''

        hints = []
        hints_ids = []
        regret = 0  
        hint = " <hint> "
        h = ""
        if gold_explanation == generated_explanation:
            hint = hint + " No "
            regret = 0
        else:
            list_eq1 = gold_explanation.split(' ')
            list_eq2 = generated_explanation.split(' ') 

            if gold_explanation.count("|")>generated_explanation.count("|"): 
                hint = hint + " add an operator. " 
                regret += 0
            
            elif generated_explanation.count("|")>gold_explanation.count("|"): 
                hint = hint + " remove an operator. "
                regret += 0
            
            if len(list_eq2)>len(list_eq1):
                difference_position = [pos for pos in range(len(list_eq1)) if list_eq2[pos] != list_eq1[pos]]
                h, regret = self.gen_hint(list_eq1, difference_position, regret)
            else: 
                difference_position = [pos for pos in range(len(list_eq2)) if list_eq2[pos] != list_eq1[pos]]
                difference_position.extend([pos for pos in range(len(list_eq2), len(list_eq1))])
                h, regret = self.gen_hint(list_eq1, difference_position, regret)
            hint +=h 
        if hint=="":
            hint = [" <hint> No" + " | EOH "]
        else:
            hint = [hint + " | EOH "]
        hints.extend(hint)
        
        return regret, hints
    

    def _critique_gen(self, critique_model, input_ids, gen_token, gen_ids, label):
        '''
        ------------------------
        Parameter: 
        generated explantion: 
        gold explanation: 
        ------------------------
        Output: 
        Hints
        '''
        
        _input = torch.cat((input_ids, gen_ids), -1)
        _input = self.move_zeros(_input).to(self.device)
        _attention = _input.clone()
        _attention[_input!=0] = 1
        _attention.to(self.device)

        h_ids, h_token = self.generate_hint(critique_model, _input, _attention)

        regret, _ = self._critique_function(gen_token, label) 
        return regret, h_token

    def gen_hint(self, equation, difference_position, regret):
        hint = ""
        operation_list = ['add', 'subtract', 'divide','multiply']
        number_list = ['number0', 'number1', 'number2', 'number3', 'number4', 'number5', 'number6', '#0', '#1', 'number1,','number2,', 'number0,', 'number3,', '#0,', '#1,']
        #print(difference_position)
        for index in difference_position: 
            if equation[index] in operation_list:
                if index <7:  
                    hint = hint + " the operator in #"+ str(0)+ " is incorrect. "
                    regret += 0
                elif index >=7 and index <14:  
                    hint = hint + " the operator in #"+ str(1)+ " is incorrect. "
                    regret += 0
                elif index >=14 and index <21:  
                    hint = hint + " the operator in #"+ str(2)+ " is incorrect. "
                    regret += 0
                else: 
                    hint = hint + " the operator in #"+ str(3)+ " is incorrect. "
                    regret += 0
            elif equation[index] in number_list:
                if index <7:
                    if index==3:   
                        hint = hint + " the first number in #"+ str(0)+ " is incorrect. "
                    elif index==4:
                        hint = hint + " the second number in #"+ str(0)+ " is incorrect. "
                    regret +=0
                elif index >=7 and index <14:
                    if index==10:  
                        hint = hint + " the first number in #"+ str(1)+ " is incorrect. "
                    else:
                        hint = hint + " the second number in #"+ str(1)+ " is incorrect. "
                    regret +=0
                elif index >=14 and index <21:  
                    if index==17: 
                        hint = hint + " the first number in #"+ str(2)+ " is incorrect. "
                        
                    else:
                        hint = hint + " the second number in #"+ str(2)+ " is incorrect. "
                    regret += 0
                else:
                    if index==25:  
                        hint = hint + " the first number in #"+ str(3)+ " is incorrect. "
                    else:
                        hint = hint + " the second number in #"+ str(3)+ " is incorrect. "
                    regret += 0

        return hint, regret

    def _predict(self,
                 eval_dataset,
                 model,
                 per_gpu_eval_batch_size,
                 max_generated_tokens):

        eval_batch_size = per_gpu_eval_batch_size * max(1, self.n_gpu)

        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        model.to(self.device)
        # multi-gpu eval
        if self.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", eval_batch_size)
        preds = []
        
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            with torch.no_grad():
                if self.n_gpu > 1:
                    outs = model.module.generate(input_ids=batch[0].cuda(),
                                            attention_mask=batch[1].cuda(),
                                            max_length=max_generated_tokens, 
                                            do_sample=False, 
                                            num_beams=1,
                                            top_k=0)
                else:
                    outs = model.generate(input_ids=batch[0].cuda(),
                                          attention_mask=batch[1].cuda(),
                                          max_length=max_generated_tokens,
                                          num_beams=1, 
                                          do_sample=False,
                                          top_k=0)
                
                dec = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
                preds.extend(dec)

        return preds

    def _tokenids2tokens(self, 
                    token_ids):
        tokens = []
        for ids in token_ids: 
            pred = [self.tokenizer.decode(ids, skip_special_tokens=True)]
            tokens.append(pred[0])
        
        return tokens

    def batch_move_zeros(self, 
                attention_mask):
        
        y = torch.empty(0, attention_mask.size(1)).to(self.device)
        for r in attention_mask:
            nz = r.nonzero().squeeze().to(self.device)
            z = torch.zeros(r.numel() - nz.numel()).to(self.device)
            z = torch.cat((r[nz], z)).unsqueeze(0)
            y = torch.cat((y, z))

        return y.to(torch.long)
    
    def move_zeros(self, 
                attention_mask):
        
        y = torch.empty(0, len(attention_mask)).to(self.device)
        nz = attention_mask.nonzero().squeeze().to(self.device)
        z = torch.zeros(attention_mask.numel() - nz.numel()).to(self.device)
        z = torch.cat((attention_mask[nz], z)).unsqueeze(0)
        y = torch.cat((y, z))

        return y.to(torch.long)

    def _add_gen_input(self, gen, input_ids, label):

        inputs = self.move_zeros(input_ids).to(self.device)
        
        attention = inputs.clone()
        attention[inputs!=0] = 1
        attention.to(self.device)
        gen_input = {"input_ids": inputs, "attention_mask": attention, "labels": label}
        
        return gen_input

    def get_sample(self,
                    input_ids,
                    attention_mask,    
                    labels,
                    model,
                    max_generated_tokens,
                    exploration_number):
            
            model.to(self.device)
            num_return_sequences = 3
            # multi-gpu eval
            if self.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
                model = torch.nn.DataParallel(model)

            # Eval!
            preds = []
            ids = []
            max_length = labels.size(1)+1
            with torch.no_grad():
                if self.n_gpu > 1:
                    outs = model.module.generate(input_ids=input_ids.cuda(),
                                            attention_mask=attention_mask.cuda(),
                                            max_length = max_generated_tokens,
                                            early_stopping = False, 
                                            return_dict_in_generate=True,
                                            do_sample = True,
                                            top_p = 0.5,
                                            num_return_sequences = exploration_number)
                else:
                    outs = model.generate(input_ids=input_ids.cuda(),
                                        attention_mask = attention_mask.cuda(),
                                        max_length = max_generated_tokens,
                                        return_dict_in_generate=True,
                                        do_sample = True,
                                        output_scores = True, 
                                        top_p = 0.5, 
                                        num_return_sequences = exploration_number)
                
            loss_fct = torch.nn.CrossEntropyLoss()
            gen_sequences = outs.sequences
            dec = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs.sequences]
                 
            id_sequence = [ids for ids in outs.sequences]
            ids.extend(id_sequence)
            preds.extend(dec)
            
            return  ids, preds











