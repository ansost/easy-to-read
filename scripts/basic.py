"""Calculate some dependency-based/POS-tag measures for the training data.

Usage:
  basic.py (--data=<data> | -d <data>)

Options:
  -d <data>, --data <data>     Select the dataset to compute the measures for ("train", "trial").
"""

import spacy
import docopt
from docopt import docopt
import pandas as pd
from tqdm.auto import tqdm
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


def counts(doc) -> dict[str, int]:
    """Count disjunctions, conjunctions and verbs in a sentence."""
    disj = 0
    conj = 0
    verbs = 0
    token_count = 0

    for token in doc:
        if token.dep_ == "disj":
            disj += 1
        if token.dep_ == "conj":
            conj += 1
        if token.pos_ == "VERB":
            verbs += 1
        token_count += 1

    return {
        "disjunctions": disj,
        "conjunctions": conj,
        "verbs": verbs,
        "tokens": token_count,
    }


def classification(phrase) -> str:
    """Classify a phrase."""
    classi = pipe(phrase)[0]
    classi["score"] = round(classi["score"], 2)
    return classi


def preprocess(phrase) -> dict[str, int]:
    """Preprocess a phrase."""
    measures = {}
    doc = nlp(phrase)
    measures.update(dependency_length(doc))
    measures.update(counts(doc))
    measures.update(classification(phrase))
    return measures


if __name__ == "__main__":
    args = docopt(__doc__)
    nlp = spacy.load("de_dep_news_trf")  # or news
    pipe = pipeline(
        "text-classification", model="krupper/text-complexity-classification"
    )

    df = pd.read_csv(f"../data/{args['--data']}.csv")
    df = df.assign(**df["phrase"].progress_apply(preprocess).apply(pd.Series))
    df.to_csv(f"../data/metrics/{args['--data']}_basics.csv", index=False)

    stats = round(df.describe(percentiles=[]), 2)
    stats = stats.drop(["count", "50%"])
    stats = stats.reindex(["min", "max", "mean", "std"])
    stats.to_csv(f"../data/metrics/{args['--data']}_basics_stats.csv")
