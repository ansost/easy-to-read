"""Generate annotation tables from spaCy dependency parses."""

import numpy as np
import pandas as pd
import spacy


def annotate_sentence(model, sentence):
    """Parse a sentence and return a DataFrame of token annotations."""
    doc = model(sentence)
    annotations = pd.DataFrame(
        [[token.text, token.i, token.pos_, token.dep_, token.head.i] for token in doc],
        columns=["token", "token_index", "POS", "deprel", "head_index"],
    )
    return annotations


if __name__ == "__main__":
    model = spacy.load("de_core_news_sm")
    sentence = "Der Abfall kommt in Tonnen oder Säcke."
    annotations = annotate_sentence(model, sentence)
    print(annotations)
    doc = model(sentence)
    spacy.displacy.render(doc, style="dep")
