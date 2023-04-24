import argparse
from tqdm import tqdm
import pathlib

from utils import (
    read_json, write_json, SITUATION_TOKEN, INTENTION_TOKEN, MORAL_ACTION_TOKEN, IMMORAL_ACTION_TOKEN, NORM_TOKEN
)

NO_HINT = "no hint"
CONTRADICTION_HINT = "contradiction"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Path to moral stories dataset")
    parser.add_argument("--suffix", type=str, default="", help="Optional file suffix")

    args = parser.parse_args()

    data = read_json(args.datapath)
    critic_data = []

    for sample in tqdm(data, total=len(data), desc="Preparing"):
        situation = sample["situation"]
        intention = sample["intention"]
        immoral_action = sample["immoral_action"]
        moral_action = sample["moral_action"]
        gold_norm = sample["norm"]
        anti_norm = sample["anti_norm"]
        other_norms = sample["other_norms"]
        other_anti_norms = sample["other_anti_norms"]
        context = f"{SITUATION_TOKEN} {situation} {INTENTION_TOKEN} {intention} {IMMORAL_ACTION_TOKEN} {immoral_action} {NORM_TOKEN} {{norm}} {MORAL_ACTION_TOKEN} {moral_action}"

        critic_data.append({
            "id": sample["id"],
            "critic_input": context.format(norm=gold_norm.capitalize()),
            "critic_output": NO_HINT,
            "gold_norm": gold_norm
        }) 

        if anti_norm:
            critic_data.append({
                "id": sample["id"],
                "critic_input": context.format(norm=anti_norm.capitalize()),
                "critic_output": CONTRADICTION_HINT,
                "gold_norm": gold_norm
            })

        for o_norm in other_norms:
            critic_data.append({
                "id": sample["id"],
                "critic_input": context.format(norm=o_norm.capitalize()),
                "critic_output": NO_HINT,
                "gold_norm": gold_norm
            })

        for a_norm in other_anti_norms:
            critic_data.append({
                "id": sample["id"],
                "critic_input": context.format(norm=a_norm.capitalize()),
                "critic_output": CONTRADICTION_HINT,
                "gold_norm": gold_norm
            })

        norm_action = sample["norm_action"] if sample["norm_action"] else gold_norm
        norm_sentiment = sample["norm_sentiment"]
        fake_norms = sample["fake_norms"]
        fake_norm_sentiments = sample["fake_norm_sentiments"]

        for fake_norm, fn_sentiment in zip(fake_norms[:-1], fake_norm_sentiments[:-1]):
            hint = norm_action.strip()
            
            if norm_sentiment * fn_sentiment < 0:
                if hint.lower().startswith("not"):
                    hint = hint[3:].strip()
                else:
                    hint = f"not {hint}"

            critic_data.append({
                "id": sample["id"],
                "critic_input": context.format(norm=fake_norm.capitalize()),
                "critic_output": hint.lower(),
                "gold_norm": gold_norm
            })

    datapath = pathlib.Path(args.datapath)
    write_json(critic_data, f"{datapath.parent}/{datapath.stem}_final{args.suffix}.json")

if __name__ == "__main__":
    main()