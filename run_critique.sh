export PYTHONPATH="${PYTHONPATH}:/home/paul/gpt-j-generation"
echo "equation_generation"
python3 src/scripts/train_predict.py --training-file data/mwp/critique_train.json --validation-file data/mwp/critique_val.json --language-model google/flan-t5-base --model-dir flan_t5_large_model --critique_model-dir output_critique  --epochs 10 --batch-size 2 --number_turn 4
