"""Extract AMR and dependency tree features, build feature matrices, and run classifiers.

Pipeline:
1. Data loading and augmentation (round-trip translation)
2. AMR parsing via English translation
3. Prolog input/output for AMR and dependency tree features
4. Feature matrix construction
5. Additional NLP features (POS counts, constituency tree counts)
6. Classifier experiments (Random Forest, SVM, MLP, Decision Tree, Logistic Regression)
"""

import ast
import argparse

import numpy as np
import pandas as pd
import spacy
from deep_translator import GoogleTranslator
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier


# --- Configuration ---

DATA_PATH = "data/"
METRICS_PATH = "data/metrics/"
TEMP_PATH = "data/temp/"
PROLOG_PATH = "data/prologData/"


# --- Data Augmentation ---

def augment_by_translation(df, source_lang="de", pivot_lang="fi", min_statements=1):
    """Augment data via round-trip translation through a pivot language."""
    df_aug = df[df["num_statements"] > min_statements].copy()
    df_aug["pivot"] = df_aug["phrase"].apply(
        lambda x: GoogleTranslator(source=source_lang, target=pivot_lang).translate(x)
    )
    df_aug["phrase"] = df_aug["pivot"].apply(
        lambda x: GoogleTranslator(source=pivot_lang, target=source_lang).translate(x)
    )
    df_aug.drop(columns=["pivot"], inplace=True)
    return df_aug


def create_augmented_data(data_path=DATA_PATH):
    """Create augmented dataset from Finnish and Chinese round-trip translations."""
    df = pd.read_csv(f"{data_path}train.csv")
    df_trial = pd.read_csv(f"{data_path}trial.csv")
    df = pd.concat([df, df_trial], join="inner")
    df = df.drop_duplicates(subset=["sent-id"], keep="last")

    # Finnish augmentation (statements > 1)
    df_fin = augment_by_translation(df, pivot_lang="fi", min_statements=1)
    df_fin["sent-id"] = df_fin["sent-id"].apply(lambda x: x + 1000000)
    df_fin.to_csv(f"{data_path}temp/finnish_augm.csv", index=False)

    # Chinese augmentation (statements > 2)
    df_zh = augment_by_translation(df, pivot_lang="zh-CN", min_statements=2)
    df_zh["sent-id"] = df_zh["sent-id"].apply(lambda x: x + 10000000)
    df_zh.to_csv(f"{data_path}temp/chinese_aug.csv", index=False)

    augmented = pd.concat([df_fin, df_zh], join="inner")
    augmented.to_csv(f"{data_path}augmented.csv", index=False)
    print(f"Augmented data shape: {augmented.shape}")
    return augmented


# --- AMR Processing ---

def translate_and_parse_amr(df, data_path=DATA_PATH):
    """Translate German text to English and compute AMR graphs."""
    import amrlib
    amrlib.setup_spacy_extension()
    nlp = spacy.load("en_core_web_sm")

    df["english"] = df["phrase"].apply(
        lambda x: GoogleTranslator(source="auto", target="en").translate(x)
    )
    amr = []
    for text in df["english"]:
        doc = nlp(text)
        graphs = doc._.to_amr()
        amr.append(graphs)
    df["amr"] = amr
    return df


def transform_graph(g):
    """Transform AMR graph string to Prolog-compatible nested list format."""
    g = list(ast.literal_eval(g))
    g = str(g[0].split("\n(")[1:])
    g = "(" + g
    g = g.replace("polarity -", "polarity negative")
    g = g.replace("polarity +", "polarity positive")
    for char in ".!?;":
        g = g.replace(char, "")
    g = g.replace(": ", "")
    g = g.replace("'", "")
    g = g.replace("[", "").replace("]", "")
    g = g.replace("/", ":instance")
    g = g.replace("\newline", "").replace("\\n", "").replace("\\", "")
    g = " ".join(g.split())
    g = g.replace("(", "[").replace(")", "]")
    g = g.replace(" ", ", ")
    g = g.replace("-", "_")
    g = g.replace(":", "attr-")
    g = g.replace(",,", ",")
    g = g.lower()
    return g


def write_prolog_amr(df, output_file):
    """Write AMR graphs as Prolog facts."""
    df["amr_prolog"] = [transform_graph(str(x)) for x in df["amr"]]
    with open(output_file, "w") as f:
        for _, row in df.iterrows():
            f.write(f"ex({row['sent-id']}, {row['amr_prolog']}).\n")
    return df


# --- Dependency Tree Processing ---

def output_to_prolog_deptree(f, sent_id, doc):
    """Write dependency tree tokens as Prolog facts."""
    for token in doc:
        f.write(
            f"item({sent_id}, token({token.head.idx}, "
            f"'{token.head.text}', '{token.dep_}', "
            f"{token.idx}, '{token.text}', '{token.pos_}')).\n"
        )


def write_prolog_deptree(df, output_file, nlp):
    """Write dependency trees for all sentences as Prolog facts."""
    with open(output_file, "w") as f:
        for _, row in df.iterrows():
            text = row["phrase"].replace("'", "")
            doc = nlp(text)
            output_to_prolog_deptree(f, row["sent-id"], doc)


# --- Feature Matrix Construction ---

def read_features_from_file(filepath):
    """Read Prolog output features from file (format: SentId; feat1, feat2, ...)."""
    features = {}
    with open(filepath) as f:
        for line in f:
            line = line.replace(" ", "").strip()
            if ";" in line:
                sent_id, feat = line.split(";", 1)
                if "," in feat:
                    features[sent_id] = feat.split(",")
    return features


def make_feature_matrix(features):
    """Convert features dict to a pivot table (feature matrix)."""
    formatted_data = []
    for key, value in features.items():
        for feature in value:
            formatted_data.append({"sent-id": key, "Feature": feature})
    formatted_df = pd.DataFrame(formatted_data)
    formatted_df = formatted_df.drop_duplicates(subset=["sent-id", "Feature"])
    feature_matrix = formatted_df.pivot_table(index="sent-id", columns="Feature", aggfunc=len, fill_value=0)
    feature_matrix.columns.name = None
    feature_matrix.reset_index(inplace=True)
    return feature_matrix


def build_feature_matrix(split_type, data_path=METRICS_PATH, prolog_path=PROLOG_PATH):
    """Build AMR and dependency tree feature matrices for a given split."""
    # AMR features
    amr_features = read_features_from_file(f"{prolog_path}features_{split_type}.txt")
    amr_matrix = make_feature_matrix(amr_features)

    df = pd.read_csv(f"{data_path}{split_type}_amr.csv")
    df["sent-id"] = df["sent-id"].astype(str)
    amr_matrix = amr_matrix.merge(df[["sent-id", "num_statements"]], on="sent-id")
    amr_matrix.rename(columns={"num_statements": "label"}, inplace=True)

    if split_type == "train":
        amr_matrix.to_csv(f"{data_path}{split_type}_amr_feature_matrix.csv", index=False)
    else:
        cols = pd.read_csv(f"{data_path}train_amr_feature_matrix.csv").columns
        amr_matrix = amr_matrix.reindex(columns=cols, fill_value=0)
        amr_matrix.to_csv(f"{data_path}{split_type}_amr_feature_matrix.csv", index=False)

    # Dependency tree features
    dep_features = read_features_from_file(f"{prolog_path}depTree_features_{split_type}.txt")
    dep_matrix = make_feature_matrix(dep_features)

    # Extra feature: number of dependency paths
    extra = {}
    for key, value in dep_features.items():
        extra[key] = sum(1 for v in value if not v.endswith("#"))
    ef = pd.DataFrame({"sent-id": list(extra.keys()), "num_dep_paths": list(extra.values())})
    dep_matrix = dep_matrix.merge(ef, on="sent-id")

    dep_matrix = dep_matrix.merge(df[["sent-id", "num_statements"]], on="sent-id")
    dep_matrix.rename(columns={"num_statements": "label"}, inplace=True)

    if split_type == "train":
        dep_matrix.to_csv(f"{data_path}{split_type}_depTree_feature_matrix.csv", index=False)
    else:
        cols = pd.read_csv(f"{data_path}train_depTree_feature_matrix.csv").columns
        dep_matrix = dep_matrix.reindex(columns=cols, fill_value=0)
        dep_matrix.to_csv(f"{data_path}{split_type}_depTree_feature_matrix.csv", index=False)

    return amr_matrix, dep_matrix


# --- Additional Features ---

def compute_additional_features(split_type, data_path=DATA_PATH, metrics_path=METRICS_PATH,
                                 temp_path=TEMP_PATH):
    """Compute POS counts, constituency tree counts, and other features."""
    nlp = spacy.load("en_core_web_sm")

    if split_type == "train":
        df_basic = pd.read_csv(f"{metrics_path}train_basics.csv")
        df_basic_trial = pd.read_csv(f"{metrics_path}trial_basics.csv")
        df_benepar = pd.read_csv(f"{metrics_path}benepar_features_train.csv")
        df_benepar_trial = pd.read_csv(f"{metrics_path}benepar_features_trial.csv")
        df_basic["tree"] = df_benepar["tree"]
        df_basic_trial["tree"] = df_benepar_trial["tree"]
        df_basic = pd.concat([df_basic, df_basic_trial], join="inner")
        df = pd.read_csv(f"{data_path}train.csv")
        df_trial = pd.read_csv(f"{data_path}trial.csv")
        df = pd.concat([df, df_trial], join="inner")
    else:
        df_basic = pd.read_csv(f"{metrics_path}{split_type}_basics.csv")
        df_benepar = pd.read_csv(f"{metrics_path}benepar_features_{split_type}.csv")
        df_basic["tree"] = df_benepar["tree"]
        df = pd.read_csv(f"{data_path}{split_type}.csv")

    df = df[["sent-id", "num_statements"]]
    common_cols = df.columns.intersection(df_basic.columns).drop("sent-id")
    df = df.drop(columns=common_cols)
    df = df.merge(df_basic, on="sent-id")

    df_add = pd.read_csv(f"{temp_path}{split_type}_temp_features.csv")
    df = df.merge(df_add, on="sent-id")
    df = df.drop_duplicates(subset=["sent-id"], keep="last")
    df = df[df["num_statements"] > 0]

    df["doc"] = df["phrase"].apply(nlp)

    # POS counts
    pos_tags = ["NOUN", "VERB", "ADJ", "ADV", "PUNCT", "ADP", "CONJ", "DET", "PRON", "NUM",
                "PART", "SYM", "AUX"]
    for pos in pos_tags:
        col_name = f"num_{pos.lower()}s" if pos not in ("AUX", "DET", "NUM", "PART", "SYM") else f"num_{pos.lower()}"
        df[col_name] = df["doc"].apply(lambda x, p=pos: sum(1 for t in x if t.pos_ == p))

    df["mean_chars_per_word"] = df["phrase"].apply(len) / df["doc"].apply(len)
    df["num_root"] = df["doc"].apply(lambda x: sum(1 for t in x if t.dep_ == "ROOT"))
    df["num_compound"] = df["doc"].apply(lambda x: sum(1 for t in x if t.dep_ == "compound"))
    df["num_root_childs"] = df["doc"].apply(lambda x: sum(1 for t in x if t.head.dep_ == "ROOT"))

    # Constituency tree counts
    for label in ["S", "VP", "NP", "PP", "ADJP", "ADVP"]:
        df[f"num_{label}"] = df["tree"].apply(lambda x: x.count(f"({label}"))

    df.to_csv(f"{metrics_path}{split_type}_additional.csv", index=False)
    return df


# --- Classifier Experiments ---

def read_feature_matrix(df_main, dflist):
    """Merge multiple feature DataFrames with main data on sent-id."""
    feature_matrix = df_main[["sent-id", "num_statements"]]
    for df in dflist:
        columns = df.columns[df.columns.isin(feature_matrix.columns)]
        columns = columns[columns != "sent-id"]
        df = df.drop(columns=columns, axis=1)
        feature_matrix = feature_matrix.merge(df, on="sent-id", how="left")
    feature_matrix = feature_matrix[feature_matrix["num_statements"] > 0]
    feature_matrix = feature_matrix.drop_duplicates(subset=["sent-id"], keep="last")
    return feature_matrix


def drop_columns(df, col_names):
    """Drop columns if they exist."""
    common = df.columns.intersection(col_names)
    return df.drop(columns=common)


def run_classifier(clf, X_train, y_train, X_test, y_test, name="Classifier"):
    """Train and evaluate a classifier."""
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred, zero_division=1))
    y_pred_train = clf.predict(X_train)
    print(f"Train Accuracy: {accuracy_score(y_train, y_pred_train):.2f}")
    return clf, y_pred


def run_all_classifiers(data_path=DATA_PATH, metrics_path=METRICS_PATH):
    """Run classifier experiments with all feature combinations."""
    # Load data
    train = pd.read_csv(f"{data_path}train.csv")
    trial = pd.read_csv(f"{data_path}trial.csv")
    train = pd.concat([train, trial], join="inner")
    train_amr = pd.read_csv(f"{metrics_path}train_amr_feature_matrix.csv")
    train_dep = pd.read_csv(f"{metrics_path}train_depTree_feature_matrix.csv")
    train_add = pd.read_csv(f"{metrics_path}train_additional.csv")

    augmented = pd.read_csv(f"{data_path}augmented.csv")
    aug_amr = pd.read_csv(f"{metrics_path}augmented_amr_feature_matrix.csv")
    aug_dep = pd.read_csv(f"{metrics_path}augmented_depTree_feature_matrix.csv")
    aug_add = pd.read_csv(f"{metrics_path}augmented_additional.csv")

    test = pd.read_csv(f"{data_path}test.csv")
    test_amr = pd.read_csv(f"{metrics_path}test_amr_feature_matrix.csv")
    test_dep = pd.read_csv(f"{metrics_path}test_depTree_feature_matrix.csv")
    test_add = pd.read_csv(f"{metrics_path}test_additional.csv")

    # Build matrices
    cols_to_drop = ["genre", "timestamp", "user", "statement_spans", "Unnamed: 1",
                    "sentence", "tree", "topic", "phrase", "phrase_number",
                    "phrase_tokenized", "pos_onehot", "doc", "label"]

    matrix_train = read_feature_matrix(train, [train_amr, train_dep, train_add])
    matrix_train = drop_columns(matrix_train, cols_to_drop)
    matrix_train = matrix_train.loc[:, ~matrix_train.columns.str.contains("^Unnamed")]

    matrix_aug = read_feature_matrix(augmented, [aug_amr, aug_dep, aug_add])
    matrix_aug = drop_columns(matrix_aug, cols_to_drop)
    matrix_aug = matrix_aug.loc[:, ~matrix_aug.columns.str.contains("^Unnamed")]

    matrix_test = read_feature_matrix(test, [test_amr, test_dep, test_add])
    matrix_test = drop_columns(matrix_test, cols_to_drop)
    matrix_test = matrix_test.loc[:, ~matrix_test.columns.str.contains("^Unnamed")]

    matrix_train_all = pd.concat([matrix_train, matrix_aug])

    # Filter low-frequency features
    min_nonzero = 4
    matrix_train_all = matrix_train_all.loc[:, (matrix_train_all != 0).sum() >= min_nonzero]
    matrix_test = matrix_test[matrix_train_all.columns]

    # Prepare X, y
    X_train = matrix_train_all.drop(["sent-id", "num_statements"], axis=1)
    y_train = matrix_train_all["num_statements"]
    X_test = matrix_test.drop(["sent-id", "num_statements"], axis=1)
    y_test = matrix_test["num_statements"]

    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

    # Scale features
    scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # 3-class labels: 1, 2, 3+
    y_train_3 = y_train.apply(lambda x: 1 if x == 1 else (2 if x == 2 else 3))
    y_test_3 = y_test.apply(lambda x: 1 if x == 1 else (2 if x == 2 else 3))

    print(f"\nLabel distribution (train): {y_train_3.value_counts().to_dict()}")
    print(f"Label distribution (test): {y_test_3.value_counts().to_dict()}")

    # Run classifiers
    classifiers = [
        ("Random Forest", RandomForestClassifier(random_state=42)),
        ("MLP", MLPClassifier(hidden_layer_sizes=(64, 16), max_iter=300, random_state=42)),
        ("SVM", SVC()),
        ("Logistic Regression", LogisticRegression(max_iter=1000)),
        ("Dummy (most frequent)", DummyClassifier(strategy="most_frequent")),
    ]

    for name, clf in classifiers:
        run_classifier(clf, X_train, y_train_3, X_test, y_test_3, name)

    # Feature importance from Random Forest
    rf_clf = RandomForestClassifier(random_state=42)
    rf_clf.fit(X_train, y_train_3)
    importances = rf_clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("\n--- Top 25 Features ---")
    for f in range(min(25, len(indices))):
        print(f"{f+1}. {X_train.columns[indices[f]]} ({importances[indices[f]]:.4f})")

    # Decision Tree (interpretable)
    top_25 = X_train.columns[indices[:25]]
    dt_clf = DecisionTreeClassifier(
        random_state=42, criterion="gini", max_depth=5, min_samples_split=20, min_samples_leaf=20
    )
    run_classifier(dt_clf, X_train[top_25], y_train_3, X_test[top_25], y_test_3, "Decision Tree (top 25)")
    print(export_text(dt_clf, feature_names=list(top_25)))


# --- AMR + Dep Tree Pipeline ---

def run_feature_pipeline(split_type, data_path=DATA_PATH, metrics_path=METRICS_PATH,
                          prolog_path=PROLOG_PATH):
    """Run the full feature extraction pipeline for a given split."""
    nlp_de = spacy.load("de_core_news_sm")

    # Load data
    df = pd.read_csv(f"{data_path}{split_type}.csv")
    if split_type == "train":
        df_trial = pd.read_csv(f"{data_path}trial.csv")
        df = pd.concat([df, df_trial], join="inner")
        df = df.drop_duplicates(subset=["sent-id"], keep="last")

    # AMR
    df = translate_and_parse_amr(df)
    df = df[df["num_statements"] > 0]
    df.to_csv(f"{metrics_path}{split_type}_amr.csv")

    # Prolog files
    write_prolog_amr(df, f"{prolog_path}{split_type}_for_prolog_amr.pl")
    write_prolog_deptree(df, f"{prolog_path}{split_type}_prolog_deptree.pl", nlp_de)

    # AMR length as temp feature
    df["amr_length"] = [len(list(ast.literal_eval(g))) for g in df["amr"]]
    df[["sent-id", "amr_length"]].to_csv(f"{data_path}temp/{split_type}_temp_features.csv")

    print(f"Pipeline complete for {split_type}. Run Prolog scripts, then call build_feature_matrix().")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMR & Dependency Tree Feature Pipeline")
    parser.add_argument("--action", choices=["pipeline", "features", "additional", "classify", "augment"],
                        required=True)
    parser.add_argument("--split", default="train", help="Data split: train, test, augmented")
    args = parser.parse_args()

    if args.action == "augment":
        create_augmented_data()
    elif args.action == "pipeline":
        run_feature_pipeline(args.split)
    elif args.action == "features":
        build_feature_matrix(args.split)
    elif args.action == "additional":
        compute_additional_features(args.split)
    elif args.action == "classify":
        run_all_classifiers()
