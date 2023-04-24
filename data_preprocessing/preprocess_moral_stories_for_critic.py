import argparse
from tqdm import tqdm
import pathlib
import spacy
import random
from itertools import product
import string

from utils import (
    read_jsonl, read_json, write_json, get_phrases, PhraseConfig, get_adjectives, get_synonyms_antonyms,
    GOOD_NORM_PREFIXES, BAD_NORM_PREFIXES, SYN_ANT_MAP
)

def _replace_word(words, word_pair):
    new_words = words.copy()

    for i, word in enumerate(words):
        if word.lower().strip().strip(string.punctuation) == word_pair[0]:
            new_words[i] = new_words[i].replace(word_pair[0], word_pair[1])
    
    return new_words

def _split_by_prefix(text):
    text = text.lower().strip()

    for p_index, prefix_options in enumerate(BAD_NORM_PREFIXES):
        for prefix in prefix_options:
            if text.startswith(prefix):
                return prefix, text.replace(prefix, "").strip(), [BAD_NORM_PREFIXES[i][random.choice(list(range(len(BAD_NORM_PREFIXES[i]))))] for i in range(len(BAD_NORM_PREFIXES)) if i != p_index]

    for p_index, prefix_options in enumerate(GOOD_NORM_PREFIXES):
        for prefix in prefix_options:
            if text.startswith(prefix):
                return prefix, text.replace(prefix, "").strip(), [GOOD_NORM_PREFIXES[i][random.choice(list(range(len(GOOD_NORM_PREFIXES[i]))))] for i in range(len(GOOD_NORM_PREFIXES)) if i != p_index]

    return None, None, None

def _choose_random_prefix(prefixes):
    p_index = random.choice(list(range(len(prefixes))))
    return prefixes[p_index][random.choice(list(range(len(prefixes[p_index]))))]

def _estimate_sentiment(text):
    for prefix_options in BAD_NORM_PREFIXES:
        for prefix in prefix_options:
            if text.startswith(prefix):
                return -1

    for prefix_options in GOOD_NORM_PREFIXES:
        for prefix in prefix_options:
            if text.startswith(prefix):
                return 1
    
    return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Path to moral stories dataset")
    parser.add_argument("--anti-ms-datapath", type=str, help="Path to contrastive moral stories dataset")
    parser.add_argument("--anti-ms-splits-datapath", type=str, help="Path to contrastive moral stories dataset with norm splits")
    parser.add_argument("--actor-datapath", type=str, help="Path to actor output dataset")
    parser.add_argument("--suffix", type=str, default="", help="File suffix")

    args = parser.parse_args()

    data = read_jsonl(args.datapath)
    actor_data = read_json(args.actor_datapath) if args.actor_datapath else None
    anti_ms_data = read_jsonl(args.anti_ms_datapath)
    anti_ms_splits_data = read_jsonl(args.anti_ms_splits_datapath)
    critic_data = []

    anti_ms_data_cache = {}

    for ams_sample in anti_ms_data:
        anti_ms_data_cache[ams_sample["ID"]] = ams_sample

    anti_ms_splits_data_cache = {}

    for ams_sample in anti_ms_splits_data:
        anti_ms_splits_data_cache[ams_sample["ID"]] = ams_sample

    for sample_index, sample in tqdm(enumerate(data), total=len(data), desc="Preprocessing"):
        anti_ms_sample = anti_ms_data_cache.get(sample["ID"])
        anti_ms_splits_sample = anti_ms_splits_data_cache.get(sample["ID"])
        actor_sample = actor_data[sample_index] if actor_data else None
        situation = sample["situation"]
        intention = sample["intention"]
        moral_action = sample["moral_action"]
        immoral_action = sample["immoral_action"]
        norm = sample["norm"]
        anti_norm =  anti_ms_sample["norm"] if anti_ms_sample else None
        norm_judgment = anti_ms_splits_sample["rot-judgment"] if anti_ms_splits_sample else None
        norm_action = anti_ms_splits_sample["rot-action"] if anti_ms_splits_sample else None
        norm_sentiment = anti_ms_splits_sample["action-moral-judgment"] if anti_ms_splits_sample else 0
        context = f"{situation} {intention} {moral_action} {immoral_action}"
        phrase_config = PhraseConfig(longest_only=False, include_det=False, include_prt=False)
        context_phrases = get_phrases(context, "vp", phrase_config=phrase_config)

        if norm_action:
            norm_phrases = get_phrases(norm_action, "vp", phrase_config=phrase_config)
        else:
            norm_phrases = get_phrases(norm, "vp", phrase_config=phrase_config)

        fake_norms = []
        fake_norm_phrases = []
        fake_norm_sentiments = []

        long_context_phrases = get_phrases(context, "vp")

        for phrase in long_context_phrases:
            judgment_choice = random.choice(["good", "bad"])
            norm_prefixes = GOOD_NORM_PREFIXES if judgment_choice == "good" else BAD_NORM_PREFIXES
            fake_norm = f"{_choose_random_prefix(norm_prefixes)} {phrase}"
            fake_norms.append(fake_norm)
            fake_norm_phrases.append(list(get_phrases(phrase, "vp", phrase_config=phrase_config)))
            fake_norm_sentiments.append(1 if judgment_choice == "good" else -1)
        
        if actor_sample:
            fake_norms.append(actor_sample["prediction"])
            fake_norm_phrases.append(list(get_phrases(actor_sample["prediction"], "vp", phrase_config=phrase_config)))
            fake_norm_sentiments.append(_estimate_sentiment(actor_sample["prediction"]))

        other_norms = []
        other_anti_norms = []

        norm_prefix, norm_suffix, norm_alt_prefixes = _split_by_prefix(norm)
        anti_norm_prefix, anti_norm_suffix, anti_norm_alt_prefixes = _split_by_prefix(anti_norm) if anti_norm else (None, None, None)

        if norm_prefix:
            for alt_prefix in norm_alt_prefixes:
                other_norms.append(f"{alt_prefix.capitalize()} {norm_suffix}")

        if anti_norm_prefix:
            for alt_prefix in anti_norm_alt_prefixes:
                other_anti_norms.append(f"{alt_prefix.capitalize()} {anti_norm_suffix}")

        norm_adjectives = get_adjectives(norm_suffix if norm_suffix else norm)
        adj_syn_pairs = [[(adj, adj)] for adj in norm_adjectives]

        for i, adj in enumerate(norm_adjectives):
            if not adj in SYN_ANT_MAP:
                SYN_ANT_MAP[adj] = get_synonyms_antonyms(adj)

            for syn in SYN_ANT_MAP[adj][0]:
                adj_syn_pairs[i].append((adj, syn))
            
        norm_words = norm.split(" ")

        for adj_comb in product(*adj_syn_pairs):
            other_norm_words = norm_words.copy()

            for adj_pair in adj_comb:
                other_norm_words = _replace_word(other_norm_words, adj_pair)

            other_norm = " ".join(other_norm_words)
            
            if other_norm.lower() != norm.lower():
                other_norms.append(other_norm)
        
        for adj in norm_adjectives:
            ants = SYN_ANT_MAP[adj][1]

            for ant in ants:
                other_anti_norm_words = _replace_word(norm_words.copy(), (adj, ant))

                other_anti_norm = " ".join(other_anti_norm_words)
                other_anti_norms.append(other_anti_norm)

        critic_data.append({
            "id": sample["ID"],
            "situation": situation,
            "intention": intention,
            "moral_action": moral_action,
            "immoral_action": immoral_action,
            "norm": norm,
            "norm_sentiment": norm_sentiment,
            "anti_norm": anti_norm,
            "norm_judgment": norm_judgment,
            "norm_action": norm_action,
            "other_norms": other_norms,
            "other_anti_norms": other_anti_norms,
            "fake_norms": fake_norms,
            "fake_norm_sentiments": fake_norm_sentiments,
            "norm_concepts": list(norm_phrases),
            "fake_norm_concepts": fake_norm_phrases,
            "context_concepts": list(context_phrases)
        })

    datapath = pathlib.Path(args.datapath)
    dataset_type = "train"

    if "dev" in datapath.stem or "val" in datapath.stem:
        dataset_type = "dev"
    elif "test" in datapath.stem:
        dataset_type = "test"

    write_json(critic_data, f"{datapath.parent}/critic_{dataset_type}_prep{args.suffix}.json")

if __name__ == "__main__":
    main()