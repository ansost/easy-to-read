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
    text = "Die Mitarbeiter reparieren gemeinsam kaputte Dinge.	"
    assert count_statements(text) == 2
