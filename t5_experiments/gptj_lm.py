import logging
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup, T5ForConditionalGeneration, T5Tokenizer, T5Config
)

from transformers import AutoModel, AutoTokenizer, GPT2Config
from transformers import GPTJForCausalLM, AutoTokenizer
from transformers import GPT2Tokenizer

from t5_experiments.data_processing.multi_task_batch_scheduler import BatchSchedulerSampler
from t5_experiments.data_processing.processor import load_and_cache_examples
from t5_experiments.data_processing.utils import get_encoded_code_tokens
from t5_experiments.eval.conala_eval import calculate_bleu_from_lists

logger = logging.getLogger(__name__)


class T5LMClassifier:
    def __init__(self,
                 max_seq_length,
                 output_model_dir,
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

        self.config = GPT2Config() 
        self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=self.pretrained_model_name_or_path,
                                                     do_lower_case=do_lower_case,
                                                     cache_dir=self.cache_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

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
        train_dataset, _ = load_and_cache_examples(data_file=training_file, local_rank=self.local_rank,
                                                   max_seq_length=self.max_seq_length, tokenizer=self.tokenizer,
                                                   evaluate=False)
        if noisy_file:
            noisy_dataset, _ = load_and_cache_examples(data_file=noisy_file, local_rank=self.local_rank,
                                                       max_seq_length=self.max_seq_length, tokenizer=self.tokenizer,
                                                       evaluate=False)

        val_dataset, val_labels = load_and_cache_examples(data_file=dev_file, local_rank=self.local_rank,
                                                          max_seq_length=self.max_seq_length, tokenizer=self.tokenizer,
                                                          evaluate=False)
        train_sampler = RandomSampler(train_dataset) if self.local_rank == -1 else DistributedSampler(train_dataset)
        if noisy_file:
            train_dataset = ConcatDataset([train_dataset, noisy_dataset])
            # dataloader with BatchSchedulerSampler
            train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                     sampler=BatchSchedulerSampler(dataset=train_dataset,
                                                                                   batch_size=per_gpu_train_batch_size),
                                                     batch_size=per_gpu_train_batch_size,
                                                     )
        else:
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)
        t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

        model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
            from_tf=bool(".ckpt" in self.pretrained_model_name_or_path),
            config=self.config,
            cache_dir=self.cache_dir,
        )
        
        model.to(self.device)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        if optimizer_algorithm == 'adam':
            optimizer = Adam(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        else:
            optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
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

        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(num_train_epochs), desc="Epoch", disable=self.local_rank not in [-1, 0]
        )
        save_steps = 500#len(train_dataset) // (per_gpu_train_batch_size * gradient_accumulation_steps* self.n_gpu)
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=self.local_rank not in [-1, 0])
            for step, batch in enumerate(epoch_iterator):
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                model.train()
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1]}#, "labels": batch[2]}#, 'label_mask': batch[3]}
                outputs = model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

                if self.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                if self.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % gradient_accumulation_steps == 0:
                    if self.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if self.local_rank in [-1, 0] and save_steps > 0 and global_step % save_steps == 0:
                        # Log metrics
                        if self.local_rank == -1:  # Only evaluate when single GPU otherwise metrics may not avg well
                            preds = self._predict(eval_dataset=val_dataset,
                                                  per_gpu_eval_batch_size=per_gpu_train_batch_size,
                                                  model=model,
                                                  max_generated_tokens=48)
                            labels = [' '.join(get_encoded_code_tokens(label)) for label in val_labels]
                            bleu, exact = calculate_bleu_from_lists(gold_texts=labels,
                                                           predicted_texts=preds)
                            print(exact, bleu)
                            if bleu > val_bleu:
                                model_to_save = (
                                    model.module if hasattr(model, "module") else model
                                )  # Take care of distributed/parallel training
                                model_to_save.save_pretrained(self.output_model_dir)
                                print('bleu on dev set improved:', bleu, ' saving model to disk.')
                                val_bleu = bleu
                            else:
                                print('bleu on dev set did not improve:', bleu)

        return global_step, tr_loss / global_step

    def predict(self,
                test_file,
                per_gpu_eval_batch_size, max_generated_tokens):
        eval_dataset, _ = load_and_cache_examples(test_file, local_rank=self.local_rank,
                                                  max_seq_length=self.max_seq_length, tokenizer=self.tokenizer,
                                                  evaluate=True)
        model = transformers.GPT2LMHeadModel(self.output_model_dir)
        return self._predict(eval_dataset=eval_dataset,
                             per_gpu_eval_batch_size=per_gpu_eval_batch_size,
                             model=model,
                             max_generated_tokens=max_generated_tokens)

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
                                            max_length=max_generated_tokens)
                else:
                    outs = model.generate(input_ids=batch[0].cuda(),
                                          attention_mask=batch[1].cuda(),
                                          max_length=max_generated_tokens)
                dec = [self.tokenizer.decode(ids) for ids in outs]
                preds.extend(dec)
                # outputs = model(**inputs)
        return preds
