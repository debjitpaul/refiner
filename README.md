# REFINER: Reasoning Feedback on Intermediate Representations :rocket: 

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![MIT License](https://img.shields.io/github/license/m43/focal-loss-against-heuristics)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2210.07228-b31b1b.svg)](https://arxiv.org/pdf/2304.01904.pdf)


Official implementation of 📖 [REFINER: Reasoning Feedback on Intermediate Representations](https://arxiv.org/pdf/2304.01904.pdf)


This repo proposes REFINER, an interaction-based framework for natural language reasoning tasks 🔥. REFINER is a framework that refines LMs reasoning capabilities through feedback. Our work is the first to investigate how interacting with fine-grained reasoning feedback on intermediate reasoning steps impacts the performance of LMs on reasoning tasks.

![Image](https://github.com/debjitpaul/refiner/blob/main/data/Figure1-motivational_example.gif)

## Getting started

## 🔍 Contents

- [🌟 Overview](#-overview)
- [🌟 Method](#-method)
- [🔥 Dependencies](#-dependencies)
- [🔥 Setup](#-setup)
- [🔥Data](#-data)
- [🚩Citation ](#-citation)

### Overview 

### Method 

### Dependencies

- compatible with python 3.8
- dependencies can be installed using `requirements.txt`

### Setup

Start by cloning the repository:
```bash
git clone git@github.com:debjitpaul/refiner.git
```


Install VirtualEnv using the following (optional):

```shell
$ [sudo] pip install virtualenv
```

Create and activate your virtual environment (optional):

```shell
$ virtualenv -p python3 venv
$ source venv/bin/activate
```

Install all the required packages:

```shell
$ pip install -r requirements.txt
```

### Data 

| Data                       | Reference                                                    | Output  | Description                                                  |
| :-------------------------- | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| Math Word Problem           | [📖](https://arxiv.org/pdf/2103.07191.pdf) , [🗂️](https://github.com/arkilpatel/SVAMP/tree/main/data/mawps-asdiv-a_svamp_without_questions), [🔗](https://github.com/arkilpatel/SVAMP) | Math Equations (z) and Answers (y) | Generate an equation given a math word problem question |
| Sythethic Natural Language Reasoning          | [📖](https://crfm-helm.readthedocs.io/en/latest/) , [🗂️](https://github.com/stanford-crfm/helm), [🔗](https://github.com/stanford-crfm/helm/tree/main/src/helm/benchmark/scenarios) | Reasoning steps (z) and Conclusion (y) | This task requires the model to perform deductive reasoning and generate intermediate reasoning steps z and conclusions y using closed-world rules and facts. |
| Moral Stories           | [📖](https://aclanthology.org/2021.emnlp-main.54.pdf) , [🗂️](https://tinyurl.com/moral-stories-data), [🔗](https://huggingface.co/datasets/demelin/moral_stories) | Moral Norm (z) and Moral Action (y) | Given a context x consisting of a situation, an intention, and an immoral action, the model needs to generate the moral norm z and the moral action y |


### For Supervised Instruction Finetuning Steps: 
1. [Train a Generator model without Critic in the loop (Warm Start).](#Train_Generator)
2. [Train a Critic model with negative instances and feedbacks.](#Train_Crtiic)
3. [Train the warm start generator model with critic in the loop. For training we used oracle critic.](#Refiner_Training)
4. [Inference using trained critic model in the loop.](#Inference)

### Do you have challenge finetuning REFINER with LLMs? 
5. [Training REFINER with LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS (LORA)](#Refiner_Training_with_Lora) [📖](https://arxiv.org/pdf/2106.09685.pdf).

### Baseline Train PPO:
Paper: [📖](https://arxiv.org/abs/2210.01241)| Code: [🔗](https://rl4lms.apps.allenai.org/)
  

#### 1. Train Generator

```
python3 src/scripts/finetune.py --training-file path_train_data --validation-file path_val_data --language-model google/flan-t5-base --model-dir flan_t5_large_model  --epochs 10 --batch-size 8
```
#### 2. Train Critic
```
python3 src/scripts/finetune.py --training-file path_train_data --validation-file path_val_data --language-model google/flan-t5-base --model-dir flan_t5_large_model --epochs 10 --batch-size 8
```
#### 3. Train REFINER 
```
python3 src/scripts/train_refiner.py --training-file data/mwp/critique_train.json --validation-file data/mwp/critique_val.json --language-model google/flan-t5-base --model-dir flan_t5_large_model --critique_model-dir output_critique  --epochs 10 --batch-size 8 --number_turn 4
```
#### 4. REFINER Inference
```
python3 src/scripts/test_predict.py --training-file data/mwp/critique_train.json --validation-file data/mwp/critique_val.json --language-model google/flan-t5-base --model-dir flan_t5_large_model --critique_model-dir output_critique  --epochs 10 --batch-size 8 --number_turn 4
```
#### 5. Train REFINER with Lora 
```
python3 src/scripts/test_predict.py --training-file data/mwp/critique_train.json --validation-file data/mwp/critique_val.json --language-model google/flan-t5-base --model-dir flan_t5_large_model --critique_model-dir output_critique --lora True --epochs 10 --batch-size 8 --number_turn 4
```

## Citation

```
@article{paul2023refiner,
  title={REFINER: Reasoning Feedback on Intermediate Representations},
  author={Paul, Debjit and Ismayilzada, Mete and Peyrard, Maxime and Borges, Beatriz and Bosselut, Antoine and West, Robert and Faltings, Boi},
  journal={arXiv preprint arXiv:2304.01904},
  year={2023}
}
```

