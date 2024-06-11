"""Calculate some dependency-based/POS-tag measures for the training/trial data.

Usage:
  python basic.py --dataset <dataset>
"""

import spacy
import argparse
import numpy as np
import pandas as pd
from typing import Generator
from tqdm.auto import tqdm
from spacy.glossary import GLOSSARY
from textstat import textstat
from transformers import pipeline

tqdm.pandas()


def dependency_length(doc: str) -> dict[str, float]:
    """Return max, min, and avg. dependency length of a sentence."""
    lengths = []
    for token in doc:
        if token.head != token:
            lengths.append(abs(token.head.i - token.i))

    return {
        "mean_dep_length": round(sum(lengths) / len(lengths), 2),
        "max_dep_length": max(lengths),
        "min_dep_length": min(lengths),
    }


def mean_dep_length(phrase: str) -> float:
    """Return the mean dependency length of a sentence."""
    return dependency_length(phrase)["mean_dep_length"]


def max_dep_length(phrase: str) -> int:
    """Return the maximum dependency length of a sentence."""
    return dependency_length(phrase)["max_dep_length"]


def min_dep_length(phrase: str) -> int:
    """Return the minimum dependency length of a sentence."""
    return dependency_length(phrase)["min_dep_length"]


def pos_onehot(doc: str) -> dict[str, int]:
    """Return one-hot encoded POS tags of a sentence.
    The vector always corresponds to the same order as the keys of the GLOSSARY."""
    pos_onehot_vector = {pos: 0 for pos in GLOSSARY.keys()}
    for token in doc:
        pos_onehot_vector[token.pos_] = 1
    arr = list(pos_onehot_vector.values())
    return arr


def counts(doc: str) -> dict[str, int]:
    """Count disjunctions, conjunctions and vesrbs in a sentence."""
    disj = 0
    conj = 0
    verbs = 0

    for token in doc:
        if token.dep_ == "disj":
            disj += 1
        if token.dep_ == "conj":
            conj += 1
        if token.pos_ == "VERB":
            verbs += 1

    return disj, conj, verbs


def verb_count(phrase) -> int:
    """Count the number of verbs in a sentence."""
    return counts(phrase)[2]


def disj_count(phrase) -> int:
    """Count the number of disjunctions in a sentence."""
    return counts(phrase)[0]


def conj_count(phrase) -> int:
    """Count the number of conjunctions in a sentence."""
    return counts(phrase)[1]


def token_count(phrase) -> int:
    """Count the number of tokens in a sentence."""
    return len(phrase)


def class_score(phrase) -> str:
    """Classify a phrase."""
    classi = pipe(phrase)[0]
    classi["score"] = np.round(classi["score"], 2)
    return classi["score"]


def flesh_reading_ease(phrase: str) -> float:
    """Return the Flesch reading ease score of a sentence."""
    return textstat.flesch_reading_ease(phrase)


def get_funcs(family: str) -> Generator[callable, None, None]:
    if family == "spacy":
        measure_funcs = {
            pos_onehot,
            mean_dep_length,
            max_dep_length,
            min_dep_length,
            token_count,
            verb_count,
            disj_count,
            conj_count,
        }

    if family == "other":
        measure_funcs = {
            class_score,
            flesh_reading_ease,
        }
    for measure in measure_funcs:
        yield measure


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        help="Only the train dataset is supported.",
    )
    args = parser.parse_args()

    nlp = spacy.load("de_dep_news_trf")
    pipe = pipeline(
        "text-classification", model="krupper/text-complexity-classification"
    )

    df = pd.read_csv(f"../data/{args.dataset}.csv")
    df["spacy_phrase"] = df["phrase"].progress_apply(lambda x: nlp(x))

    for measure in tqdm(get_funcs(family="spacy")):
        df[measure.__name__] = df["spacy_phrase"].apply(measure)
    df.drop("spacy_phrase", axis=1, inplace=True)

    for measure in tqdm(get_funcs(family="other")):
        df[measure.__name__] = df["phrase"].apply(measure)

    benepar_res = pd.read_csv(f"../data/bracket_is_sent_results_{args.dataset}.csv")
    df["is_sent"] = benepar_res["is_sent"]
    df["big_np_count"] = benepar_res["big_np_count"]
    df["big_pp_count"] = benepar_res["big_pp_count"]

    print(df.head())
    df.to_csv(f"../data/metrics/{args.dataset}_basics.csv", index=False)

    stats = round(df.describe(percentiles=[]), 2)
    stats = stats.drop(["count", "50%"])
    stats = stats.reindex(["min", "max", "mean", "std"])
    stats.to_csv(f"../data/metrics/{args.dataset}_basics_stats.csv")
