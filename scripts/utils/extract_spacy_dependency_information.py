"""Extract dependency information from spaCy parses in multiple output formats.

Provides functions to export token dependency information as:
- List of lists
- JSON object
- Prolog arguments
"""

import spacy
from spacy import displacy


def output_to_list_of_lists(doc):
    """Return a list of lists with token text, POS tag, dependency label, and head token text."""
    return [[token.text, token.pos_, token.dep_, token.head.text] for token in doc]


def output_to_json_object(doc):
    """Return a dict mapping token text to its POS, dependency, and head."""
    return {
        token.text: {"POS": token.pos_, "Dependency": token.dep_, "Head": token.head.text}
        for token in doc
    }


def output_to_prolog_argument(doc):
    """Print Prolog-formatted token facts."""
    for token in doc:
        print(f"token('{token.text}', '{token.pos_}', '{token.dep_}', '{token.head.text}').")


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("Credit and mortgage account holders must submit their requests")

    displacy.render(doc, style="dep", jupyter=False, options={"distance": 100})

    for token in doc:
        print(f"Token: {token.text}, POS: {token.pos_}, Dependency: {token.dep_}, Head: {token.head.text}")

    print("\nList of lists:")
    print(output_to_list_of_lists(doc))

    print("\nJSON object:")
    print(output_to_json_object(doc))

    print("\nProlog arguments:")
    output_to_prolog_argument(doc)
