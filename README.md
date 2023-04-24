# REFINER: Reasoning Feedback on Intermediate Representations :rocket:
Official implementation of [REFINER: Reasoning Feedback on Intermediate Representations](https://arxiv.org/pdf/2304.01904.pdf)


This repo proposes REFINER, an interaction-based framework for natural language reasoning tasks 🔥. REFINER is a framework that refines LMs reasoning capabilities through feedback. Our work is the first to investigate how interacting with fine-grained reasoning feedback on intermediate reasoning steps impacts the performance of LMs on reasoning tasks.

## Getting started 

**Training**

```
python3 src/scripts/train_predict.py --training-file data/mwp/critique_train.json --validation-file data/mwp/critique_val.json --language-model google/flan-t5-base --model-dir flan_t5_large_model --critique_model-dir output_critique  --epochs 10 --batch-size 8 --number_turn 4
```
**Inference**


## Citation

```
s@article{paul2023refiner,
  title={REFINER: Reasoning Feedback on Intermediate Representations},
  author={Paul, Debjit and Ismayilzada, Mete and Peyrard, Maxime and Borges, Beatriz and Bosselut, Antoine and West, Robert and Faltings, Boi},
  journal={arXiv preprint arXiv:2304.01904},
  year={2023}
}
```

