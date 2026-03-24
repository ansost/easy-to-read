"""Constituency parsing with benepar for statement detection and NP/PP analysis."""

import pandas as pd
import benepar
import spacy
from nltk import Tree
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix


# --- Parsing ---

def setup_parser(lang="de"):
    """Load spaCy + benepar pipeline for German or English."""
    if lang == "de":
        nlp = spacy.load("de_dep_news_trf")
        model_name = "benepar_de2"
    else:
        nlp = spacy.load("en_core_web_md")
        model_name = "benepar_en3"

    if spacy.__version__.startswith("2"):
        nlp.add_pipe(benepar.BeneparComponent(model_name))
    else:
        nlp.add_pipe("benepar", config={"model": model_name})
    return nlp


def parse_sent_w_spacy(nlp, sent):
    """Parse a sentence and determine if it contains a statement (S + V pattern)."""
    doc = nlp(sent)
    sent_obj = list(doc.sents)[0]
    toks = [token.text for token in doc]

    if "(S " in sent_obj._.parse_string and "(V" in sent_obj._.parse_string:
        is_sent = 1
    elif len(toks) < 3:
        is_sent = 1
    else:
        is_sent = 0

    return sent_obj._.parse_string, is_sent


def parse_sentence_pretokenized(parser, df, sent_idx):
    """Parse a pre-tokenized sentence from the dataframe."""
    sent_toks = df["phrase_tokenized"][sent_idx].split()
    sent_toks = [s.split(":=")[1] for s in sent_toks]
    result = parser.parse(sent_toks)
    tokens = df["phrase_tokenized"][sent_idx]
    statement_spans = df["statement_spans "][sent_idx]
    return result, tokens, statement_spans


def parse_all_sentences(nlp, df):
    """Parse all sentences and return list of (sentence, tree, is_sent) tuples."""
    results = []
    for sent in tqdm(df["phrase"]):
        tree_as_string, is_sent = parse_sent_w_spacy(nlp, sent)
        results.append((sent, tree_as_string, is_sent))
    return results


# --- Evaluation ---

def evaluate_is_sent(data_path="data/"):
    """Evaluate is_sent predictions against gold labels across train/trial/test."""
    splits = {
        "train": ("train.csv", "benepar_features_train.csv"),
        "trial": ("trial.csv", "benepar_features_trial.csv"),
        "test": ("test.csv", "benepar_features_test.csv"),
    }

    gold_all, pred_all = [], []
    for split_name, (data_file, annotation_file) in splits.items():
        df = pd.read_csv(f"{data_path}{data_file}")
        gold = [num > 0 for num in df["num_statements"]]
        annotations = pd.read_csv(f"{data_path}metrics/{annotation_file}")
        pred = list(annotations["is_sent"])
        gold_all.extend(gold)
        pred_all.extend(pred)

    accuracy = accuracy_score(gold_all, pred_all)
    cm = confusion_matrix(gold_all, pred_all)
    print(f"Accuracy: {accuracy}")
    print(pd.DataFrame(cm))
    return accuracy, cm


# --- NP/PP analysis ---

def np_pp_size_count(tree_strs):
    """Count NP and PP subtrees and their sizes from parse tree strings."""
    np_count_sizes = []
    pp_count_sizes = []

    for tree_str in tree_strs:
        t = Tree.fromstring(tree_str)

        # NP subtrees
        np_subtrees = [s for s in t.subtrees() if s.label() == "NP"]
        num_np = len(np_subtrees)
        np_sizes = [len(s.leaves()) for s in np_subtrees] if num_np else [0]
        np_count_sizes.append((num_np, np_sizes))

        # PP subtrees
        pp_subtrees = [s for s in t.subtrees() if s.label() == "PP"]
        num_pp = len(pp_subtrees)
        pp_sizes = [len(s.leaves()) for s in pp_subtrees] if num_pp else [0]
        pp_count_sizes.append((num_pp, pp_sizes))

    count_df = pd.DataFrame(np_count_sizes, columns=["num_np", "np_sizes"])
    num_pp, pp_sizes = zip(*pp_count_sizes)
    count_df["num_pp"] = pd.Series(num_pp)
    count_df["pp_sizes"] = pd.Series(pp_sizes)
    return count_df


def compute_big_constituents(res, np_threshold=2, pp_threshold=3):
    """Count NPs bigger than threshold and PPs bigger than threshold."""
    big_np_count = [sum(1 for size in sizes if size > np_threshold) for sizes in res["np_sizes"]]
    big_pp_count = [sum(1 for size in sizes if size > pp_threshold) for sizes in res["pp_sizes"]]
    return big_np_count, big_pp_count


def extract_nps(tagged_sentence):
    """Extract noun phrases from a tagged sentence string."""
    tree = Tree.fromstring(tagged_sentence)
    nps = []
    for subtree in tree.subtrees():
        if subtree.label() == "NN":
            nps.append(" ".join(subtree.leaves()))
    return nps


def extract_constituents_with_sizes(tagged_sentence, labels=None):
    """Extract constituents matching given labels and return (text, size) tuples."""
    if labels is None:
        labels = {"NN", "NNS", "NNP", "NNPS"}
    tree = Tree.fromstring(tagged_sentence)
    results = []
    for subtree in tree.subtrees():
        if subtree.label() in labels:
            text = " ".join(subtree.leaves())
            results.append((text, len(subtree.leaves())))
    return results


if __name__ == "__main__":
    nlp = setup_parser("de")

    # Parse example
    sent = "Hier ist ein gutes Beispiel für einen einfachen Satz"
    tree_str, is_sent = parse_sent_w_spacy(nlp, sent)
    print(tree_str, is_sent)

    # Parse dataset
    df = pd.read_csv("data/trial.csv")
    results = parse_all_sentences(nlp, df)
    df_results = pd.DataFrame(results, columns=["sentence", "tree", "is_sent"])
    df_results.to_csv("data/bracket_is_sent_results.csv", index=False)
    print(f"Parsed {len(df_results)} sentences")

    # Evaluate
    evaluate_is_sent()
