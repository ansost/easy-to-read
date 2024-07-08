"""Parse a phrase and return the number of statements according to the guidelines. 
"""

import pandas as pd

#def check_date_entity(line, date_entity_pattern):
#    return bool(date_entity_pattern.search(line))


#def load_amr(text, split):
#    splits_dict = {
#        "train": "../data/metrics/train_amr.csv",
#        "test": "../data/metrics/test_amr.csv",
#        "eval": "../data/metrics/eval_amr.csv",
#    }
#    df = pd.read_csv(splits_dict[split])
#    return df[df["phrase"] == text]["amr"].values[0]


def count_statements(
    text: str,
    nlp,
    special_cases_pattern,
    sein_verbforms,
) -> int:
    
    doc = nlp(text)
    statements = 0

    #if check_date_entity(amr, check_date_entity):
        #statements += 1  # Date specifications

    for sent in doc.sents:
        if special_cases_pattern.match(sent.text):
            statements += 1  # Handle special cases
            continue

        clauses = []
        current_clause = []
        has_sein_statement = False

        for token in sent:
            if token.text in ["(", ")"]:
                statements += 1  # Parentheses

            if (
                token.pos_ == "ADJ"
                and token.head.pos_ == "NOUN"
                and token.text not in current_clause
            ):
                statements += 1  # Separating via single adjectives

            if token.dep_ == "pobj" and token.head.pos_ == "ADP":
                statements += 1  # Separating via prepositional phrase

            if (
                token.dep_ == "pobj"
                and token.head.pos_ == "ADP"
                and len(token.text.split()) == 1
            ):
                statements -= 1  # Composites

            if (
                token.dep_ == "pobj"
                and token.head.pos_ == "ADP"
                and token.head.head.pos_ == "VERB"
            ):
                statements -= 1  # Trivial prepositions

            if token.ent_type_ == "DATE":
                statements += 1  # Date and year specifications

            current_clause.append(token.text)

            if token.pos_ == "VERB" and token.morph.get("VerbForm") == ["Fin"]:
                clauses.append(current_clause)
                current_clause = []

            if token.text in sein_verbforms:
                has_sein_statement = True

        if current_clause:
            clauses.append(current_clause)

        for clause in clauses:
            clause_doc = nlp(" ".join(clause))
            has_subject = any(token.dep_ in ["sb", "nk", "pg"] for token in clause_doc)
            has_verb = any(token.pos_ == "VERB" for token in clause_doc)

            if has_subject and has_verb:
                statements += 1  # SVO combination forming a statement
            elif has_verb:
                statements += 1  # Subclause with a verb forming a statement

        if not any(token.pos_ == "VERB" for token in sent) and not has_sein_statement:
            statements = 0  # 0-statement sentences

    return statements