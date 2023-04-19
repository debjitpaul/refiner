import logging
from tokenizers import Token
import torch
torch.manual_seed(123)
from torch import nn
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import AutoModelForCausalLM 
from torch.nn import CrossEntropyLoss
import wandb
import torch.nn.functional as F
wandb.init(project="dual_learning_AC", entity="debjitpaul")

import random
random.seed(123)

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup, T5ForConditionalGeneration, T5Tokenizer, T5Config
)

from t5_experiments.data_processing.multi_task_batch_scheduler import BatchSchedulerSampler
from t5_experiments.data_processing.processor import load_and_cache_examples
from t5_experiments.data_processing.utils import get_encoded_code_tokens
from t5_experiments.eval.conala_eval import calculate_bleu_from_lists
from accelerate import Accelerator
accelerator = Accelerator()

logger = logging.getLogger(__name__)

import torch.distributed as dist

#dist.init_process_group(backend='smddp')

class T5LMClassifier:
    def __init__(self,
                 max_seq_length,
                 output_model_dir,
                 output_critique_model, 
                 pretrained_model_name_or_path,
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
            pretrained_model_name_or_path="t5-base",
            from_tf=bool(".ckpt" in self.pretrained_model_name_or_path),
            config=self.config,
            cache_dir=self.cache_dir,
        )
        
        #model = T5ForConditionalGeneration.from_pretrained(self.output_model_dir)
        critique_model = T5ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path="t5-base",
            from_tf=bool(".ckpt" in self.pretrained_model_name_or_path),
            config=self.config,
            cache_dir=self.cache_dir,
        )#T5ForConditionalGeneration.from_pretrained(self.output_critique_model)
        model.to(self.device)
        critique_model.to(self.device)

        per_gpu_eval_batch_size =8 

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

        critique_optimizer_grouped_parameters = [
            {
                "params": [p for n, p in critique_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {"params": [p for n, p in critique_model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]

        if optimizer_algorithm == 'adam':
            policy_optimizer = Adam(policy_optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
            critique_optimizer = Adam(critique_optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        else:
            policy_optimizer = AdamW(policy_optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
            critique_optimizer = AdamW(critique_optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        
        policy_scheduler = get_linear_schedule_with_warmup(
            policy_optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )

        critique_scheduler = get_linear_schedule_with_warmup(
            critique_optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
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
            critique_model = torch.nn.DataParallel(critique_model)

        # Distributed training (should be after apex fp16 initialization)
        if self.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True
            )
            critique_model = torch.nn.parallel.DistributedDataParallel(
                critique_model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True
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
        critique_model.zero_grad()
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
                num_turn = 3
                turn_loss = 0
                loss = 0
                reward = 0
                penalty = 1
                count = 0
                prev_hint_ids = []
                prev_token_ids = []
                
                for turn in range(1, num_turn):
                    outputs = model(**inputs) # equation generation model 
                    turn_loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
                    
                    count+=1
                    # generate equation 
                    token_ids, tokens, sample_loss, greedy_loss = self.get_sample(input_ids=_input, 
                            attention_mask=_attention,
                            labels = _labels,
                            model=model,
                            max_generated_tokens=50) 
                    labels = self._tokenids2tokens(batch[2])
                    
                    c = 0 
                    regret = 0
                    h_best = ["No"]
                    hints = []
                    regret_batch = 0 
                    best_tokens = []
                    r = 0
                    t = ""
                    explore_loss = 0
                    best_loss = 0
                    
                    for i in range(len(_input)): # number of sequences = #batch_size * gradient_accumulation_steps
                        explore_loss = 0
                        regret = 1000
                        # oracle critique
                        #r , h_tok = self._critique_gen(critique_model, _input[i], tokens[c], token_ids[c], labels[i])
                        #print(tokens, labels)
                        r , h_tok = self._critique_function(tokens[i], labels[i])
                        greedy_loss[i] = greedy_loss[i]*r
                        #if r <= regret:
                        h_best = h_tok
                        regret = r
                        t = tokens[i]
                        #else: 
                        h_best = h_best
                         
                        best_loss += explore_loss 
                        best_tokens.extend([t])
                        regret_batch += regret
                        hints.extend(h_best) 

                    regret_batch = regret_batch/(len(_input))
                    hint_ids = self.tokenizer.batch_encode_plus(hints, padding=True, return_tensors="pt").input_ids.to(self.device)
                    token_ids = self.tokenizer.batch_encode_plus(best_tokens, padding=True, return_tensors="pt").input_ids.to(self.device) 
                    
                    
                    _input = torch.cat((batch[0], token_ids, hint_ids), 1)
                    
                    _input = self.batch_move_zeros(_input).to(self.device)
                    _attention = _input.clone()
                    _attention[_input!=0] = 1
                    _attention.to(self.device)

                    prev_token_ids = token_ids
                    prev_hint_ids = hint_ids

                    inputs = {"input_ids": _input, "attention_mask": _attention, "labels": batch[2]}
                    labels = self._tokenids2tokens(batch[2])
                    if turn<2: 
                        prev_logits = outputs[1]
                    else: 
                         
                    '''
                    true_hint_ids, true_hints = self._critique_function_batch(best_tokens, labels)
                    critique_input = torch.cat((batch[0], token_ids), 1)
                    critique_input = self.batch_move_zeros(critique_input).to(self.device)
                    critique_attention = critique_input.clone()
                    critique_attention[critique_input!=0] = 1
                    critique_attention.to(self.device)
                    critique_inputs = {"input_ids": critique_input, "attention_mask": critique_attention, "labels": true_hint_ids}
                    critique_model.to(self.device)
                    critique_model.train()
                    critique_loss = critique_model(**critique_inputs)[0]
                    '''

                    try:
                        bleu, exact = calculate_bleu_from_lists(gold_texts=labels,
                                                          predicted_texts=best_tokens)
                    except ZeroDivisionError:
                        bleu = 0
                        exact = 0
                    
                    # rl_loss = torch.sum(gen_loss, dim=-1)
                    # print(rl_loss)
                    # print(gen_loss)
                    
                    greedy_loss = torch.stack(greedy_loss, dim=1)#.sum(dim=0) #.sum(dim=0) 
                    rl_loss = Variable(greedy_loss.data, requires_grad=True)
                    
                    #print("RL Loss : ",greedy_loss, "\n")
                    #print(turn_loss)
                    #print(turn_loss)
                    #print(loss)
                    loss += turn_loss +  greedy_loss + (1-exact)
                    print(turn_loss, greedy_loss) 
                    #critique_loss += critique_loss 
                    if (1 - exact) == 0:
                        break

                actor_loss = loss/count
                #critic_loss = critique_loss/count
                if self.n_gpu > 1:
                    actor_loss = actor_loss.mean()  # mean() to average on multi-gpu parallel training
                    #critic_loss = critic_loss.mean()
                
                if gradient_accumulation_steps > 1:
                    actor_loss = actor_loss / gradient_accumulation_steps
                    #critic_loss = critic_loss / gradient_accumulation_steps
                
                total_loss = actor_loss #+ critic_loss 
                
                if self.fp16:
                    with amp.scale_loss(actor_loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                elif loss is not None:
                    actor_loss.backward()

                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                #policy_optimizer.step()
                #policy_scheduler.step()  # Update learning rate schedule
                #model.zero_grad()

                #critic_loss.backward()
                #torch.nn.utils.clip_grad_norm_(critique_model.parameters(), max_grad_norm)
                #critique_optimizer.step()
                #critique_scheduler.step()  # Update learning rate schedule
                #critique_model.zero_grad() 
                
                tr_loss += actor_loss.item()
                if (step + 1) % gradient_accumulation_steps == 0:
                    
                    if self.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(policy_optimizer), max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                    policy_optimizer.step()
                    policy_scheduler.step()  # Update learning rate schedule
                    
                    model.zero_grad()
                    #critique_optimizer.step()
                    #critique_scheduler.step()  # Update learning rate schedule
                    #critique_model.zero_grad() 
                    
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
                                                  max_generated_tokens=50) 
                            labels = [' '.join(get_encoded_code_tokens(label)) for label in train_labels]
                            bleu, exact = calculate_bleu_from_lists(gold_texts=labels,
                                                           predicted_texts=preds)
                            wandb.log({'train_exact_match': exact, 'train_bleu': bleu})
                            print(exact, bleu)
                            preds = self._predict(eval_dataset=val_dataset,
                                                  per_gpu_eval_batch_size=per_gpu_eval_batch_size,
                                                  model=model,
                                                  max_generated_tokens=50)
                            labels = [' '.join(get_encoded_code_tokens(label)) for label in val_labels]
                        
                            bleu, exact = calculate_bleu_from_lists(gold_texts=labels,
                                                           predicted_texts=preds)
                            
                            print(exact, bleu)
                            wandb.log({'val_exact_match': exact, 'val bleu': bleu})
                            wandb.log({'training_losses': loss})
                            if exact > val_exact:
                                model_to_save = (
                                    model.module if hasattr(model, "module") else model
                                )  # Take care of distributed/parallel training
                                model_to_save.save_pretrained(self.output_model_dir)
                                '''
                                critique_model_to_save = (
                                    critique_model.module if hasattr(model, "module") else critique_model
                                )  # Take care of distributed/parallel training
                                critique_model_to_save.save_pretrained(self.output_critique_model)
                                '''
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
        if gold_explanation == generated_explanation:
            hint = hint + " No "
            regret = 0
        else:
            list_eq1 = gold_explanation.split(' ')
            list_eq2 = generated_explanation.split(' ') 
            if len(list_eq2)>len(list_eq1):
                difference_position = [pos for pos in range(len(list_eq1)) if list_eq2[pos] != list_eq1[pos]]
                hint, regret = self.gen_hint(list_eq2, difference_position, regret)
            else: 
                difference_position = [pos for pos in range(len(list_eq2)) if list_eq2[pos] != list_eq1[pos]]
                hint, regret = self.gen_hint(list_eq1, difference_position, regret)

            if gold_explanation.count("#")>generated_explanation.count("#"): 
                hint = hint + " add an operation. " 
                regret += 1.5
            elif generated_explanation.count("#")>gold_explanation.count("#"): 
                hint = hint + " remove an operation. "
                regret += 1.5
        hint = [hint + " <hint> "]
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
        
        '''
        try:
            bleu, exact = calculate_bleu_from_lists(gold_texts=label,
                                                          predicted_texts=gen_token)
        except ZeroDivisionError:
            bleu = 0
            exact = 0
        regret = 1 - exact
        '''
        regret, _ = self._critique_function(gen_token, label) 
        return regret, h_token

    def gen_hint(self, equation, difference_position, regret):
        hint = ""
        operation_list = ['add', 'substract', 'divide','multiply']
        for index in difference_position: 
            if equation[index] in operation_list:
                hint = hint + "the operation in the position "+ str(index)+ " is incorrect. "
                regret += 2.5
            else:
                hint = hint + "the number in the position "+ str(index)+ " is incorrect. "
                regret += 6.5

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

        #inputs = torch.cat((input_ids, gen))
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
                    max_generated_tokens):
            
            model.to(self.device)
            num_return_sequences = 3
            # multi-gpu eval
            if self.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
                model = torch.nn.DataParallel(model)

            # Eval!
            preds = []
            ids = []
            max_length = labels.size(1)+1
            #with torch.no_grad():
            if self.n_gpu > 1:
                    outs = model.module.generate(input_ids=input_ids.cuda(),
                                            attention_mask=attention_mask.cuda(),
                                            max_length = max_length,
                                            early_stopping = False, 
                                            return_dict_in_generate=True,
                                            do_sample = True,
                                            #decoder_input_ids = labels, 
                                            output_scores = True,
                                            output_hidden_states=True, 
                                            top_k = 50,
                                            num_return_sequences = 1, 
                                            length_penalty=0)
            else:
                    outs = model.generate(input_ids=input_ids.cuda(),
                                        attention_mask = attention_mask.cuda(),
                                        decoder_input_ids = labels,
                                        max_length = max_generated_tokens,
                                        return_dict_in_generate=True,
                                        do_sample = True,
                                        output_scores = True, 
                                        top_k = 50,  
                                        top_p=0,
                                        temperature=0, 
                                        num_return_sequences = 5)
                
            loss_fct = torch.nn.CrossEntropyLoss()
            gen_sequences = outs.sequences
            dec = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs.sequences]
                 
            id_sequence = [ids for ids in outs.sequences]
            ids.extend(id_sequence)
            preds.extend(dec)
            
            
            #sm = torch.nn.functional.softmax(outs['scores'][0], dim=1)
                
            logits = torch.stack(outs.scores, dim=1) 
                
            # loss = loss_fct(sm.view(-1, sm.size(-1)), labels.contiguous().view(-1)) 
            # print(logits.size())
                
            #print(loss)
            #logits.requires_grad = True
            probs = 1/(1+torch.exp(-logits)) # logits --> probs
            #probs.requires_grad = True
                
            #print(probs)
            #print(probs.size())

            #loss = loss_fct(probs.view(-1, probs.size(-1)), labels.contiguous().view(-1))
                
            #print(loss)
                
            #nll = -torch.log(torch.sum(probs, dim=-1)) # negative log probs
            #print(nll)
            #print(-torch.log(probs))
            #logp = F.log_softmax(logits, dim=2)
            #print(logp.size())
            #logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
            #print(logpy.size())
            #exit()
            #nll.requires_grad = True
            #smooth_loss = torch.sum(nll, dim=-1)/nll.size(-1) # normalizing
            #print(smooth_loss.size()) 
                
            #gen_probs = torch.gather(logits, 2, gen_sequences[:, :-1, None]).squeeze(-1)
            #smooth_loss.requires_grad = True 
                
            # print(labels.contiguous().view(-1).size())
            # print(logits.view(-1, logits.size(-1)).size())
                
            #loss = None
                
            #if labels is not None:
            #    loss_fct = CrossEntropyLoss()
            #    loss = loss_fct(probs.view(-1, probs.size(-1)), labels.contiguous().view(-1)) 
                
            #print(loss)
            '''
            #exit()
            gen_probs = torch.sum(torch.nn.functional.log_softmax(logits), -1)
                #loss = torch.max(torch.nn.functional.log_softmax(gen_probs, dim=-1),1)
            print(gen_probs)
                 
            #gen_probs.requires_grad = True
            print(gen_probs.size()) 
            #gen_probs = torch.sum(gen_probs, 1)
            unique_prob_per_sequence = gen_probs.prod(-1)
            normed_gen_probs = gen_probs / gen_probs.sum(0)
            print(normed_gen_probs)
            assert normed_gen_probs[:, 0].sum() == 1.0
            print(gen_probs)
            ''' 
            sample_losses = []
            greedy_losses = []
                
            model.train()
                
            #out = model(input_ids = input_ids.cuda(), attention_mask = attention_mask.cuda(), labels = labels.cuda())
            #print(out[0])
                 
            for index in range(len(input_ids)):
                gen_inputs = self._add_gen_input(gen_sequences[index], input_ids[index], labels[index])
                label_tokens = self.tokenizer.decode(gen_sequences[index], skip_special_tokens=True) 
                label = self.tokenizer(label_tokens, return_tensors="pt").input_ids
                outputs = model(input_ids = gen_inputs['input_ids'].cuda(), attention_mask=gen_inputs['attention_mask'].cuda(), labels=label.cuda())
                sloss = Variable(outputs[0], requires_grad = True)
                sample_losses.append(sloss)
                    
                #label_tokens = self.tokenizer.decode(labels[index], skip_special_tokens=True)
                #label = self.tokenizer(label_tokens, return_tensors="pt").input_ids
                #outputs = model(input_ids = gen_inputs['input_ids'], attention_mask=gen_inputs['attention_mask'], labels=label.cuda())
                #gloss = Variable(outputs[0], requires_grad = True)
                #greedy_losses.append(gloss)
            
            return  ids, preds, sample_losses, sample_losses











