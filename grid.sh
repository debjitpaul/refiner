for lm in t5-small t5-base t5-large
do
for l in 0.0001 0.0005 0.001 0.003 0.005
do
for b in 8 16
do
for g 4 16
do
PYTHONPATH=. nohup python t5_experiments/scripts/train_predict.py --training-file data/conala-train.json --validation-file data/conala-test.json --model-dir m-$lm-$b-$g-$l --epochs 30 --batch $b --lr $l --gradient-accumulation $g --language-model $lm >results.gold-$lm-$b-$g-$l.out
done
done
done
done
