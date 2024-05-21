"""
Script to get sentences from a csv file
"""
import pandas as pd


def get_sentences(dataset: str) -> pd.DataFrame:
    """Get sentences from a csv file."""
    data = pd.read_csv(f"../data/{dataset}")
    sentences = data["phrase"].tolist()
    return sentences

def remove_newline(text: str) -> str:
    """Remove newline characters from text."""
    return text.replace("\\newline", "")

def remove_extra_spaces(text: str) -> str:
    """Remove extra spaces from text."""
    return " ".join(text.split())

if __name__ == "__main__":
    sentences = get_sentences("train_updated.csv")
    with open("../data/german_sentences.txt", "w") as file:
        for sentence in sentences:
            sentence = remove_newline(sentence)
            sentence = remove_extra_spaces(sentence)
            file.write(sentence + "\n")
