"""
Script to create train, validation, and test splits
"""
# create a script to create train, validation, and test splits with the following requirements:
# - train: 80%, validation: 10%, test: 10%
# - the script should read the sentences from the german_sentences.txt and english_sentences.txt files
# write the script in splits.py
# that is it
import pandas as pd
from sklearn.model_selection import train_test_split

def get_sentences(dataset: str) -> pd.DataFrame:
    """Get sentences from a csv file."""
    data = pd.read_csv(f"../data/{dataset}")
    sentences = data["phrase"].tolist()
    return sentences

# write split function
def split_data(data: pd.DataFrame, test_size: float, val_size: float) -> pd.DataFrame:
    """Split data into train, validation, and test sets."""
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=val_size, random_state=42)
    return train_data, val_data, test_data

if __name__ == "__main__":
    sentences = get_sentences("train_updated.csv")
    with open("../data/german_sentences.txt", "w") as file:
        for sentence in sentences:
            sentence = remove_newline(sentence)
            sentence = remove_extra_spaces(sentence)
            file.write(sentence + "\n")
    # read german sentences
    with open("../data/german_sentences.txt", "r") as file:
        german_sentences = file.readlines()
    # read english sentences
    with open("../data/english_sentences.txt", "r") as file:
        english_sentences = file.readlines()
    # create a dataframe
    data = pd.DataFrame({"german": german_sentences, "english": english_sentences})
    # split data
    train_data, val_data, test_data = split_data(data, test_size=0.1, val_size=0.1)
    # save data
    train_data.to_csv("../data/train.csv", index=False)
    val_data.to_csv("../data/val.csv", index=False)
    test_data.to_csv("../data/test.csv", index=False)
    print("Data split successfully!")
