import argparse
import os
import csv
import json

from src.eval.conala_eval import calculate_bleu_from_lists
from src.test_lm import T5LMClassifier
from src.data_processing.utils import read_labels, get_encoded_code_tokens

DATA_FOLDER = 'data'

def training(training_file, dev_file,
             trained_models_dir,
             trained_critique_dir,
             per_gpu_train_batch_size,
             learning_rate,
             epochs,
             language_model,
             grad_acc,
             sequence_length,
             optimizer_algorithm='adam',
             noisy_file=None,):

    if not os.path.exists(trained_models_dir):
        os.mkdir(trained_models_dir)
    classifier = T5LMClassifier(
            max_seq_length=sequence_length,
            output_model_dir=trained_models_dir,
            output_critique_model=trained_critique_dir, 
            cache_dir=os.path.join(DATA_FOLDER, 'pretrained'),
        pretrained_model_name_or_path=language_model
    )
    classifier.train(training_file, dev_file,
                         per_gpu_train_batch_size=per_gpu_train_batch_size,
                         learning_rate=learning_rate,
                         optimizer_algorithm=optimizer_algorithm,
                         num_train_epochs=epochs,
                     noisy_file=noisy_file,
                     gradient_accumulation_steps=grad_acc)


def evaluate(test_file, trained_models_dir, trained_critique_dir, sequence_length,
             per_gpu_eval_batch_size, language_model):
    _classifier = T5LMClassifier(max_seq_length=sequence_length,
                                 output_model_dir=trained_models_dir,
                                 output_critique_model=trained_critique_dir, 
                                 cache_dir=os.path.join(DATA_FOLDER, 'pretrained'),
                                 pretrained_model_name_or_path=language_model
                                 )

    print(trained_models_dir)
    preds = _classifier.predict(test_file=test_file,
                                per_gpu_eval_batch_size=per_gpu_eval_batch_size,
                                max_generated_tokens=sequence_length)
    
    labels = read_labels(test_file, tag='Linear_Formula')
    inputs = read_labels(test_file, tag='Body')

    labels = [l.lower() for l in labels]
    preds = [p.lower() for p in preds]
    inputs = [i for i in inputs]
    
    #labels = [' '.join(get_encoded_code_tokens(label)) for label in labels]
    new_labels = []
    
    with open(trained_models_dir+"/result.csv", 'w', encoding='UTF8', newline='') as outfile: 
        for index in range(len(labels)):
            try:
                encoded_reconstr_code = get_encoded_code_tokens(labels[index])
            except:
                print("Error related to brackets", labels[index])
                continue
            label = ' '.join(encoded_reconstr_code)
            new_labels.append(labels[index])
            print(preds[index].strip() == labels[index].strip()) 
            if preds[index].strip() == labels[index].strip():
                outfile.write(inputs[index] +'\t'+ preds[index] +'\t'+labels[index]+'\t'+ "yes" +'\n')
            else:
                outfile.write(inputs[index] +'\t'+ preds[index] +'\t'+labels[index]+'\t'+ "no" +'\n')
            
     
    eval_results = calculate_bleu_from_lists(gold_texts=new_labels, predicted_texts=preds)
    print(eval_results)

    return eval_results

def parse_args():
    parser = argparse.ArgumentParser(description='Critique T5')

    parser.add_argument('--training-file', dest='training_file', required=False, help='Path to training file',
                        default=None)
    parser.add_argument('--noisy-file', dest='noisy_file', required=False, help='Path to noisy file',
                        default=None)
    parser.add_argument('--validation-file', dest='validation_file', required=False, help='Path to validation file')
    parser.add_argument('--language-model', default='t5-base', help='Can be either some huggingface model or a '
                                                                         'path to a model. If the path is in GCS we '
                                                                         'download it first.')
    parser.add_argument('--model-dir', dest='model_dir', required=True,
                        help='the folder/google bucket in which the model will be stored or loaded from.')
    parser.add_argument('--critique_model-dir', dest='critique_model_dir', required=True,
                        help='the folder/google bucket in which the model will be stored or loaded from.')
    parser.add_argument('--epochs', default=20,
                        help='number of epochs to train')
    parser.add_argument('--batch-size', default=1,
                        help='batch size')
    parser.add_argument('--val-batch-size', default=1,
                        help='validation batch size')
    parser.add_argument('--number_turn', default=4,
                        help='learning rate')
    parser.add_argument('--lr', default=0.0001,
                        help='learning rate')
    parser.add_argument('--gradient-accumulation', default=1)
    parser.add_argument('--local_rank', default=-1)
    args = parser.parse_args()
    
    return args


def main():
    args = parse_args()
    #   train
    language_model = args.language_model
    if args.training_file and args.validation_file:
        training(training_file=args.training_file, 
                 dev_file=args.validation_file,
                 trained_models_dir=args.model_dir,
                 trained_critique_dir=args.critique_model_dir,
                 per_gpu_train_batch_size=int(args.batch_size),
                 epochs=int(args.epochs),
                 learning_rate=float(args.lr),
                 sequence_length=512,
                 noisy_file=args.noisy_file,
                 language_model=language_model,
                 grad_acc=int(args.gradient_accumulation))
    
    if args.validation_file:
            evaluation_results = evaluate(test_file=args.validation_file,
                                      trained_models_dir=args.model_dir,
                                      trained_critique_dir=args.critique_model_dir,
                                      per_gpu_eval_batch_size=int(args.val_batch_size),
                                      sequence_length=512,
                                          language_model=language_model
                                    )
if __name__ == '__main__':
    main()
