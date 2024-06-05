import spacy
import re

def get_statement_spans(text: str) -> list:
    # Load the German language model
    nlp = spacy.load("de_core_news_sm")

    text = text.replace("\\newline", "\n")

    doc = nlp(text)

    statement_spans = []
    current_span = []

    # Define articles and conjunctions to exclude
    articles = ["der", "die", "das", "ein", "eine", "einer", "eines", "einem", "einen"]
    coordinating_conjunctions = ["und", "oder", "aber", "denn", "sondern", "doch"]

    for sent in doc.sents:
        clauses = []
        current_clause = []

        for token in sent:
            if token.text in ["(", ")"]:
                if current_span:
                    statement_spans.append(current_span)
                    current_span = []

            if token.pos_ == "ADJ" and token.head.pos_ == "NOUN":
                if token.text not in current_clause:
                    if current_span:
                        statement_spans.append(current_span)
                    current_span = [token.i]

            if token.pos_ == "NUM" or token.pos_ == "INTJ" or token.pos_ == "PART":
                continue  # Quantifiers and filler words are no separate statements

            if token.pos_ == "ADJ" and token.head.pos_ == "NOUN" and (token.morph.get("Degree") == ["Cmp"] or token.morph.get("Degree") == ["Sup"]):
                continue  # Comparatives and superlatives

            if token.dep_ == "pobj" and token.head.pos_ == "ADP":
                if current_span:
                    statement_spans.append(current_span)
                current_span = [token.i]

            if token.dep_ == "pobj" and token.head.pos_ == "ADP" and len(token.text.split()) == 1:
                current_span = current_span[:-1]

            if token.dep_ == "pobj" and token.head.pos_ == "ADP" and token.head.head.pos_ == "VERB":
                current_span = current_span[:-1]

            if token.ent_type_ == "DATE":
                if current_span:
                    statement_spans.append(current_span)
                current_span = [token.i]
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
                if current_span:
                    statement_spans.append(current_span)
                current_span = [token.i]

            elif has_verb:
                if current_span:
                    statement_spans.append(current_span)
                current_span = [token.i]

        if not any(token.pos_ == "VERB" for token in sent):
            if current_span:
                statement_spans.append(current_span)

        # Remove spans contained in other spans
        statement_spans = [span for span in statement_spans if not any(set(span).issubset(other) for other in statement_spans if other != span)]

    # Filter out articles, coordinating conjunctions, and newline tokens
    filtered_spans = []
    for span in statement_spans:
        filtered_span = [i for i in span if doc[i].text.lower() not in articles + coordinating_conjunctions and doc[i].text != "\n"]
        if filtered_span:
            filtered_spans.append(filtered_span)

    return filtered_spans
