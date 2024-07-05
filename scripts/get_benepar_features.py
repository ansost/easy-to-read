"""Get features of sentences using Benepar, including:
    - bracketed parse tree
    - whether the sentence is a sentence (1 or 0)
    - number of big NPs and PPs in each sentence 

Output: ../data/metrics/benepar_features_{args.dataset}.csv with 5 columns

Usage:
    $ python get_benepar_features.py --dataset {train/trial}
"""

import argparse
import pandas as pd
import benepar, spacy
from nltk import Tree
from tqdm import tqdm


def parse_sent_w_spacy(nlp, sent):
    """Parse a sentence with spaCy and Benepar
    Return the parsed tree as a bracket string and a boolean indicating if the sentence is a sentence"""
    doc = nlp(sent)
    sent = list(doc.sents)[0]
    toks = [token.text for token in doc]

    if "(S " in sent._.parse_string and "(V" in sent._.parse_string:
        is_sent = 1
    elif len(toks) < 3:
        is_sent = 1
    else:
        is_sent = 0

    return sent._.parse_string, is_sent


def count_big_np_pp(tree_strs):
    """Count the number of big NPs and PPs in each sentence
    Big NP: NP with more than 2 words; big PP: PP with more than 3 words
    Return the counts of big NPs and PPs in all sentences as two lists"""
    # NP counts and sizes in each sentence
    np_count_sizes = []
    for tree in tree_strs:
        t = Tree.fromstring(tree)
        # find all NP subtrees
        np_subtrees = [subtree for subtree in t.subtrees() if subtree.label() == 'NP']

        num_np = len(np_subtrees)
        # if there is at least on NP, get the sizes of NP subtrees
        np_sizes = []
        if num_np:
            for subtree in np_subtrees:
                np_sizes.append(len(subtree.leaves()))
        else: 
            np_sizes.append(0)
        
        # append the count and sizes to the list
        np_count_sizes.append((num_np, np_sizes))

    # do the same for PP counts and sizes
    pp_count_sizes = []
    for tree in tree_strs:
        t = Tree.fromstring(tree)
        pp_subtrees = [subtree for subtree in t.subtrees() if subtree.label() == 'PP']

        num_pp = len(pp_subtrees)
        pp_sizes = []
        if num_pp:
            for subtree in pp_subtrees:
                pp_sizes.append(len(subtree.leaves()))
        else: 
            pp_sizes.append(0)
        
        pp_count_sizes.append((num_pp, pp_sizes))

    # create the DataFrame, first save the NP counts and sizes
    count_df = pd.DataFrame(np_count_sizes, columns=["num_np", "np_sizes"])

    num_pp, pp_sizes = zip(*pp_count_sizes)
    # add the new columns to the DataFrame
    count_df['num_pp'] = pd.Series(num_pp)
    count_df['pp_sizes'] = pd.Series(pp_sizes)

    # if NP has more than 2 words, PP has more than 3 words, count as big
    big_np_count = []
    for np_size in count_df["np_sizes"]:
        big_np_count.append(sum([1 for size in np_size if size > 2]))

    big_pp_count = []
    for pp_size in count_df["pp_sizes"]:
        big_pp_count.append(sum([1 for size in pp_size if size > 3]))

    return big_np_count, big_pp_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        help="Only the train/trial dataset is supported.",
    )
    args = parser.parse_args()

    # load the training data
    df = pd.read_csv(f'../data/{args.dataset}.csv')
    df.head()

    # load up Benepar model to spaCy pipeline
    nlp = spacy.load('de_dep_news_trf')
    if spacy.__version__.startswith('2'):
        nlp.add_pipe(benepar.BeneparComponent("benepar_de2"))
    else:
        nlp.add_pipe("benepar", config={"model": "benepar_de2"})

    # parse all sentences in the dataframe
    tree_as_string_list = []
    is_sent_list = []
    for sent in tqdm(df["phrase"]):
        tree_as_string, is_sent = parse_sent_w_spacy(nlp, sent)
        tree_as_string_list.append(tree_as_string)
        is_sent_list.append(is_sent)

    # make a copy of df, keep only needed columns
    if args.dataset == "eval":
        benepar_res = df[["sent-id", "phrase"]].copy()
    else:
        benepar_res = df[["sent-id", "phrase", "num_statements", "statement_spans"]].copy()

    benepar_res["tree"] = tree_as_string_list
    benepar_res["is_sent"] = is_sent_list
    # transform all tree strings to workable format
    tree_strs  = ["(" + tree + ")" for tree in benepar_res["tree"]]
    
    # update big NP and PP counts
    big_np_count, big_pp_count = count_big_np_pp(tree_strs)
    benepar_res["big_np_count"] = big_np_count
    benepar_res["big_pp_count"] = big_pp_count

    # print preview and save the results as metrics
    print(benepar_res.head())
    benepar_res.to_csv(f'../data/metrics/benepar_features_{args.dataset}.csv', index=False)