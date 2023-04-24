import sys
import json

sys.path.append("src")

# Clone HELM repo (https://github.com/stanford-crfm/helm) and run this script from the root directory
from src.helm.benchmark.scenarios.synthetic_reasoning_natural_scenario import SRNScenario

def get_datasets(difficulty="easy"):
    train = []
    val = []
    test = []

    srn = SRNScenario(difficulty)
    instances = srn.get_instances()

    for instance in instances:
        inputs = instance.input.split("\n")
        fact_index = inputs.index("Fact:")
        rules = inputs[:fact_index]
        fact = inputs[fact_index+1]
        question = inputs[fact_index+2]
        consequents = []
        for ref in instance.references:
            consequents.append(ref.output)
        sample = {
            "rules": rules,
            "fact": fact,
            "question": question,
            "consequents": consequents
        }
        if instance.split == "valid":
            val.append(sample)
        elif instance.split == "test":
            test.append(sample)
        else:
            train.append(sample)
    
    return train, val, test

def save_dataset(dataset, path):
    with open(path, "w") as f:
        json.dump(dataset, f, indent=2)

easy_train, easy_val, easy_test = get_datasets("easy")
medium_train, medium_val, medium_test = get_datasets("medium")
hard_train, hard_val, hard_test = get_datasets("hard")

dataset_map = {
    "easy": [easy_train, easy_val, easy_test],
    "medium": [medium_train, medium_val, medium_test],
    "hard": [hard_train, hard_val, hard_test]
}

for dataset_type, datasets in dataset_map.items():
    for dataset, split in zip(datasets, ["train", "val", "test"]):
        save_dataset(dataset, f"data/srn/{dataset_type}/{split}.json")