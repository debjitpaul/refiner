# REFINER: Reasoning Feedback on Intermediate Representations :rocket:  (EACL 2024)

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![MIT License](https://img.shields.io/github/license/m43/focal-loss-against-heuristics)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2304.01904-b31b1b.svg)](https://arxiv.org/pdf/2304.01904.pdf)


Official implementation of 📖 [REFINER: Reasoning Feedback on Intermediate Representations](https://arxiv.org/pdf/2304.01904.pdf) 🔗 [Blog Post](https://debjitpaul.github.io/refiner/)

![Image](https://github.com/debjitpaul/refiner/blob/main/data/Figure1-motivational_example.gif)

## 🔍 Contents

- [🌟 Overview](#overview)
- [🌟 Method](#method)
- [🔥 Dependencies](#dependencies)
- [🔥 Setup](#setup)
- [🔥 Data](#data)
- [🔥 Models](#models)
- [🚩 Citation ](#citation)

## Overview 

This repo proposes REFINER, an interaction-based framework for natural language reasoning tasks 🔥. REFINER is a framework that refines LMs reasoning capabilities through feedback. Our work is the first to investigate how interacting with fine-grained reasoning feedback on intermediate reasoning steps impacts the performance of LMs on reasoning tasks.

## Method 

We propose to solve these tasks by forcing the model to generate intermediate hypotheses (z) and improving them via structured feedback. We introduce an interactive framework named REFINER, made of two separate models: (a) a CRITIC model trained to provide structured feedback on intermediate reasoning steps and (b) a GENERATOR model trained to solve the reasoning task by first generating intermediate reasoning steps. The core idea of REFINER is to exploit the interaction between the generator model and the critic model, where the generator’s intermediate reasoning steps are improved via structured feedback from the critic. 

## Dependencies

- compatible with python 3.8
- dependencies can be installed using `requirements.txt`
- The codebase is built around [Hugging Face](https://huggingface.co/) ecosystem and [wandb](https://wandb.ai/site) (for monitoring and experiment management).

## Setup


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

## Data 

| Data                       | Reference                                                    | Output  | Description                                                  |
| :-------------------------- | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| Math Word Problem           | [📖](https://arxiv.org/pdf/2103.07191.pdf) , [🗂️](https://github.com/arkilpatel/SVAMP/tree/main/data/mawps-asdiv-a_svamp_without_questions), [🔗](https://github.com/arkilpatel/SVAMP) | Math Equations (z) and Answers (y) | Generate an equation given a math word problem question |
| Sythethic Natural Language Reasoning          | [📖](https://crfm-helm.readthedocs.io/en/latest/) , [🗂️](https://github.com/stanford-crfm/helm), [🔗](https://github.com/stanford-crfm/helm/tree/main/src/helm/benchmark/scenarios) | Reasoning steps (z) and Conclusion (y) | This task requires the model to perform deductive reasoning and generate intermediate reasoning steps z and conclusions y using closed-world rules and facts. |
| Moral Stories           | [📖](https://aclanthology.org/2021.emnlp-main.54.pdf) , [🗂️](https://tinyurl.com/moral-stories-data), [🔗](https://huggingface.co/datasets/demelin/moral_stories) | Moral Norm (z) and Moral Action (y) | Given a context x consisting of a situation, an intention, and an immoral action, the model needs to generate the moral norm z and the moral action y |

## Models

### Baseline
Train a baseline model using PPO.

Paper: [📖](https://arxiv.org/abs/2210.01241)| Code: [🔗](https://rl4lms.apps.allenai.org/)

### REFINER
* [Train a Generator model without Critic in the loop (Warm Start).](#train-generator)

* [Train a Critic model with negative instances and feedbacks.](#train-critic)

* [Train the warm start generator model with critic in the loop. For training we used oracle critic.](#train-refiner)

* [Inference using trained critic model in the loop.](#refiner-inference)

* [Training REFINER with Low-rank Adaptation of Large Language Models (LORA)](#train-refiner-with-lora) [📖](https://arxiv.org/pdf/2106.09685.pdf).
  

#### Train Generator

```
python3 src/scripts/finetune.py --training-file path_train_data --validation-file path_val_data --language-model google/flan-t5-base --model-dir flan_t5_large_model  --epochs 10 --batch-size 8
```
#### Train Critic
```
python3 src/scripts/finetune.py --training-file path_train_data --validation-file path_val_data --language-model google/flan-t5-base --model-dir flan_t5_large_model --epochs 10 --batch-size 8
```
#### Train REFINER 
```
python3 src/scripts/train_refiner.py --training-file data/mwp/critique_train.json --validation-file data/mwp/critique_val.json --language-model google/flan-t5-base --model-dir flan_t5_large_model --critique_model-dir output_critique  --epochs 10 --batch-size 8 --number_turn 4
```
#### REFINER Inference
```
python3 src/scripts/test_predict.py --training-file data/mwp/critique_train.json --validation-file data/mwp/critique_val.json --language-model google/flan-t5-base --model-dir flan_t5_large_model --critique_model-dir output_critique  --epochs 10 --batch-size 8 --number_turn 4
```
#### Train REFINER with Lora 
```
python3 src/scripts/test_predict.py --training-file data/mwp/critique_train.json --validation-file data/mwp/critique_val.json --language-model google/flan-t5-base --model-dir flan_t5_large_model --critique_model-dir output_critique --lora True --epochs 10 --batch-size 8 --number_turn 4
```

## Citation

```
@misc{paul2023refiner,
  title={REFINER: Reasoning Feedback on Intermediate Representations},
  author={Paul, Debjit and Ismayilzada, Mete and Peyrard, Maxime and Borges, Beatriz and Bosselut, Antoine and West, Robert and Faltings, Boi},
  eprint={2304.01904},
  journal={arXiv preprint arXiv:2304.01904},
  url={https://arxiv.org/pdf/2304.01904.pdf},
  year={2023}
}
```

