"""Visualize column distributions of the training data."""

import matplotlib.pyplot as plt
import pandas as pd


def plot_distribution(train, column, bins=100, rotation=0):
    """Plot histogram for a given column."""
    if column == "genre":
        genre = train["genre"].str.split("|", expand=True)
        genre = genre.stack().value_counts()
        genre.plot(kind="bar")
    else:
        plt.hist(train[column], bins=bins)
    if rotation:
        plt.xticks(rotation=rotation)
    plt.xlabel(column)
    plt.ylabel("count")
    plt.show()


if __name__ == "__main__":
    train = pd.read_csv("data/metrics/train_basics.csv")
    print(train.columns.tolist())

    columns_config = [
        ("max_dep_length", 100, 0),
        ("mean_dep_length", 100, 0),
        ("tokens", 100, 0),
        ("genre", 100, 0),
        ("topic", 100, 90),
        ("phrase_number", 100, 80),
        ("num_statements", 100, 0),
        ("label", 100, 0),
        ("score", 100, 0),
    ]

    for column, bins, rotation in columns_config:
        plot_distribution(train, column, bins=bins, rotation=rotation)
