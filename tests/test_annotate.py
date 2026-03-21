import pytest
import spacy
from scripts.annotate import count_statements

@pytest.fixture(scope="module")
def nlp():
    return spacy.load("de_core_news_sm")

def test_count_statements_case_2_example_1(nlp):
    text = "Die moderne Sportart heißt: Splashdiving."
    assert count_statements(text) == 2

def test_count_statements_case_2_example_2(nlp):
    text = "Der Vorschlag wurde ohne Gegenstimmen angenommen."
    assert count_statements(text) == 2

def test_count_statements_case_2_example_3(nlp):
    text = "Die Einwohner konnten ihre Lebensmittel ganz normal kaufen."
    assert count_statements(text) == 1

def test_count_statements_case_2_example_4(nlp):
    text = "Die Mitarbeiter reparieren gemeinsam kaputte Dinge."
    assert count_statements(text) == 2

def test_count_statements_case_3_example_1(nlp):
    text = "Im Jahr 1877 heiraten Alexander Graham Bell und Mabel Hubbard."
    assert count_statements(text) == 2

def test_count_statements_case_4_example_1(nlp):
    text = "An der Universität Boston."
    assert count_statements(text) == 0

def test_count_statements_case_composites(nlp):
    text = "Religion der Christen."
    assert count_statements(text) == 0

def test_count_statements_case_quantifiers(nlp):
    text = "50 Menschen"
    assert count_statements(text) == 0
