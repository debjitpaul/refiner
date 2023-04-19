export PYTHONPATH="${PYTHONPATH}:/home"
python3 t5_experiments/scripts/test_predict.py --validation-file data/critique_test.json --language-model google/flan-t5-large --model-dir flan_t5_large_model --critique_model-dir ../critique_design/critic
