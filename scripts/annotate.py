import spacy
import re


def count_statements(text: str) -> int:
    nlp = spacy.load("de_core_news_sm")
    doc = nlp(text)

    statements = 0

    coordinating_conjunctions = ["und", "oder", "aber", "denn", "sondern", "doch"]
    subordinating_conjunctions = ["dass", "ob", "weil", "da", "wenn", "als", "nachdem", "damit", "um", "so dass", "sodass", "obwohl", "obgleich", "wobei", "während", "bevor", "ehe", "seit", "seitdem", "bis", "solange", "sobald", "sooft", "wie", "als ob", "als wenn", "indem", "ohne dass", "statt dass", "anstatt dass", "außer dass", "nur dass", "kaum dass", "geschweige denn", "es sei denn", "wenn auch", "wenngleich", "gleichwohl", "trotzdem", "ungeachtet dessen"]

    for sent in doc.sents:
        clauses = []
        current_clause = []

        if re.match(r'(Das heißt|Manche sagen)', sent.text):
            statements += 1 # handle special cases
            continue

        for token in sent:
            if token.text in ["(", ")"]:
                statements += 1  # Parentheses

            if token.pos_ == "ADJ" and token.head.pos_ == "NOUN":
                if token.text not in current_clause:
                    statements += 1  # Separating via single adjectives

            if token.pos_ == "NUM" or token.pos_ == "INTJ" or token.pos_ == "PART":
                continue  # Quantifiers and filler words are no separate statements

            if token.pos_ == "ADJ" and token.head.pos_ == "NOUN" and (token.morph.get("Degree") == ["Cmp"] or token.morph.get("Degree") == ["Sup"]):
                continue  # Comparatives and superlatives

            if token.dep_ == "pobj" and token.head.pos_ == "ADP":
                statements += 1  #  Separating via prepositional phrase

            if token.dep_ == "pobj" and token.head.pos_ == "ADP" and len(token.text.split()) == 1:
                statements -= 1  # Composites

            if token.dep_ == "pobj" and token.head.pos_ == "ADP" and token.head.head.pos_ == "VERB":
                statements -= 1  # Trivial prepositions

            if token.ent_type_ == "DATE":
                statements += 1  # Date and year specifications

            current_clause.append(token.text)

            if token.pos_ == "VERB" and token.morph.get("VerbForm") == ["Fin"]:
                clauses.append(current_clause)
                current_clause = []

        if current_clause:
            clauses.append(current_clause)

        for clause in clauses:
            clause_text = " ".join(clause)
            clause_doc = nlp(clause_text)

            has_subject = any(token.dep_ in ["sb", "nk", "pg"] for token in clause_doc)
            has_object = any(token.dep_ in ["oa", "og", "op"] for token in clause_doc)
            has_verb = any(token.pos_ == "VERB" for token in clause_doc)

            if has_subject and has_verb:
                statements += 1  # SVO combination forming a statement
            elif has_verb:
                statements += 1  # Subclause with a verb forming a statement

        if not any(token.pos_ == "VERB" for token in sent):
            statements = 0  # 0-statement sentences

    return statements