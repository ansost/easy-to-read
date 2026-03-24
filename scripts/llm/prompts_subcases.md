## LLM results for subcases from the annotation guidelines

This document shows prompts and the LLM's answers for annotating whether certain subcases in the [annotation guidelines](https://german-easy-to-read.github.io/statements/annotations/) apply or not.

Testing these subcases was more experimental with hand-picked example sentences. 
More elaborate prompts and prompting methods with larger samples were used for the prediction of statements and statement spans. These can be found in ``LLAMA/api_predictions_num_statements.py`` and ``LLAMA/api_predictions_statement_spans.py``.


Cases tested:

- Identifying 0-statements
- Restrictive Adjectives/adverbs
- Prepositional phrases

Analysis was done with [Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) over [HuggingChat](https://huggingface.co/chat/)


### Identifying 0 statements
> Again, annotation guidelines and case 4 specifically were provided. 

Classify the following sentences as either containing 0 statements or more than 0 statements. For your decision, rely only on the annotation guidelines give above, specifically the section from it I have provided as case 4. Think step by step.

Some examples: 
- more than 0 statements: Alchemisten wollen etwas ver-bessern. 
- more than 0 statements: Zum Beispiel, wollen sie kranke Menschen gesund machen. 
- 0 statements: In Edinburgh, Schottland. 

Provide your output in form of a table. 

Sentences: 

Aldous Huxley kommt am 26. Juli 1894 auf die Welt.
In Godalming, Surrey.
Er stirbt am 22. November 1963. \newline In Los Angeles, USA. \newline  \newline Im Jahr 1932: \newline Huxley schreibt seinen berühmtesten Roman.
Den Science-fiction: \newline Schöne neue Welt \newline Es ist eine Dystopie.
Aldous Huxley war ein Schrift-steller.
Aleppo ist eine sehr alte Stadt.
Menschen haben hier schon 3 Tausend Jahre vor Christus gelebt.
Im Jahr 2008 haben ungefähr 1,7 Millionen Menschen in Aleppo gewohnt.
Die meisten Menschen in Aleppo sind Araber und Kurden.
Das heißt: \newline Die meisten Menschen in Aleppo sind Muslime.
In Aleppo gibt es auch Christ.
Ungefähr 15 Prozent der Menschen in Aleppo sind Christen.
Im Jahr 1952 hat ein Architekt neue Straßen geplant.


LLM output:
'''
Here is the output table with my classification of each sentence as either containing 0 statements or more than 0 statements, along with a short explanation:

| Sentence | Answer | Explanation |
| --- | --- | --- |
| Aldous Huxley kommt am 26. Juli 1894 auf die Welt. | more than 0 | Contains a verb "kommt" and a statement about Aldous Huxley's birth. |
| In Godalming, Surrey. | 0 | Lacks a verb and doesn't contain a statement. |
| Er stirbt am 22. November 1963. \newline In Los Angeles, USA. \newline  \newline Im Jahr 1932: \newline Huxley schreibt seinen berühmtesten Roman. | 0 | Multiple sentences merged into one, which is considered a 0-statement sentence. |
| Den Science-fiction: \newline Schöne neue Welt \newline Es ist eine Dystopie. | more than 0 | Contains verbs "ist" and describes a book, making it a statement. |
| Aldous Huxley war ein Schrift-steller. | more than 0 | Contains a verb "war" and a statement about Aldous Huxley's profession. |
| Aleppo ist eine sehr alte Stadt. | more than 0 | Contains a verb "ist" and a statement about Aleppo. |
| Menschen haben hier schon 3 Tausend Jahre vor Christus gelebt. | more than 0 | Contains a verb "haben" and a statement about people living in Aleppo. |
| Im Jahr 2008 haben ungefähr 1,7 Millionen Menschen in Aleppo gewohnt. | more than 0 | Contains a verb "haben" and a statement about the population of Aleppo in 2008. |
| Die meisten Menschen in Aleppo sind Araber und Kurden. | more than 0 | Contains a verb "sind" and a statement about the demographics of Aleppo. |
| Das heißt: \newline Die meisten Menschen in Aleppo sind Muslime. | more than 0 | Contains a verb "sind" and a statement about the religion of people in Aleppo. |
| In Aleppo gibt es auch Christ. | more than 0 | Contains a verb "gibt" and a statement about the presence of Christians in Aleppo. |
| Ungefähr 15 Prozent der Menschen in Aleppo sind Christen. | more than 0 | Contains a verb "sind" and a statement about the percentage of Christians in Aleppo. |
| Im Jahr 1952 hat ein Architekt neue Straßen geplant. | more than 0 | Contains a verb "hat" and a statement about an architect's plan in 1952. |

Let me know if you have any questions or need further clarification!
'''


-------
### Restrictive Adjectives and Adverbs
> In two prior prompts I gave it the annotation guidelines and case 2, so the adjective/adverbs case.

Prompt: 
For each of the sentences decide whether the adjectives/adverbs in the sentence add an additional statement or not. Multiple adjectives/adverbs can also add more than one statement to a sentence.
For your decision, rely only on the annotation guidelines give above, specifically the section from it I have provided as case 2. Think step by step. 

Some examples: 

sentence: Die moderne Sportart heißt:  Splashdiving.
answer: Adds one statement
explanation: modern gives additional information about the sport

sentence: Der Vorschlag wurde ohne Gegenstimmen angenommen.
answer: Adds one statement
explanation_ Modal adverbial ohne Gegenstimmen 

sentence:Die Einwohner konnten ihre Lebensmittel ganz normal kaufen.
answer: Does not add a statement
explanation: ganz normal gives no restrictions

sentence: Die Mitarbeiter reparieren gemeinsam kaputte Dinge.
answer: Adds one statement
explanation: Sequence: Die Mitarbeiter reparieren gemeinsam. Die Mitarbeiter reparieren kaputte Dinge.

Provide your output in form of a table detailing the sentence, your answer and a short explanation.

Here are the sentences: 

Eine sehr bekannte Alchemisten war Maria die Jüdin.
Alchemisten haben Athanoren benutzt.
Alchemie ist ein arabisches Wort.
Durch Alchemie entdeckt man neue Sachen.
Aleppo ist eine sehr alte Stadt.
Die meisten Menschen in Aleppo sind Araber und Kurden.
Sie haben viele alte Häuser abgerissen.
Und neue, moderne Wohn-blöcke gebaut.
Der Mann Ali ibn Abi Talib ist den Aleviten sehr wichtig.
Zum Beispiel tragen die meisten alevitischen Frauen kein Kopftuch.

LLMs answer: 

'''
Here is the output table with my classification of each sentence as either containing 0 statements or more than 0 statements, along with a short explanation:

| Sentence | Answer | Explanation |
| --- | --- | --- |
| Aldous Huxley kommt am 26. Juli 1894 auf die Welt. | more than 0 | Contains a verb "kommt" and a statement about Aldous Huxley's birth. |
| In Godalming, Surrey. | 0 | Lacks a verb and doesn't contain a statement. |
| Er stirbt am 22. November 1963. \newline In Los Angeles, USA. \newline  \newline Im Jahr 1932: \newline Huxley schreibt seinen berühmtesten Roman. | 0 | Multiple sentences merged into one, which is considered a 0-statement sentence. |
| Den Science-fiction: \newline Schöne neue Welt \newline Es ist eine Dystopie. | more than 0 | Contains verbs "ist" and describes a book, making it a statement. |
| Aldous Huxley war ein Schrift-steller. | more than 0 | Contains a verb "war" and a statement about Aldous Huxley's profession. |
| Aleppo ist eine sehr alte Stadt. | more than 0 | Contains a verb "ist" and a statement about Aleppo. |
| Menschen haben hier schon 3 Tausend Jahre vor Christus gelebt. | more than 0 | Contains a verb "haben" and a statement about people living in Aleppo. |
| Im Jahr 2008 haben ungefähr 1,7 Millionen Menschen in Aleppo gewohnt. | more than 0 | Contains a verb "haben" and a statement about the population of Aleppo in 2008. |
| Die meisten Menschen in Aleppo sind Araber und Kurden. | more than 0 | Contains a verb "sind" and a statement about the demographics of Aleppo. |
| Das heißt: \newline Die meisten Menschen in Aleppo sind Muslime. | more than 0 | Contains a verb "sind" and a statement about the religion of people in Aleppo. |
| In Aleppo gibt es auch Christ. | more than 0 | Contains a verb "gibt" and a statement about the presence of Christians in Aleppo. |
| Ungefähr 15 Prozent der Menschen in Aleppo sind Christen. | more than 0 | Contains a verb "sind" and a statement about the percentage of Christians in Aleppo. |
| Im Jahr 1952 hat ein Architekt neue Straßen geplant. | more than 0 | Contains a verb "hat" and a statement about an architect's plan in 1952. |

Let me know if you have any questions or need further clarification!
'''


-------

### Propositional Phrases
> Given annotation guidelines and case 3

Decide whether the propositional phrases add one or more statements to this sentence.
For your decision, rely only on the annotation guidelines give above, specifically the section from it I have provided as case 3. Think step by step. 


Provide your output in form of a table. 

Here are the sentences: 
Viele Menschen auf der ganzen Welt haben Angst und sind traurig.
Viele Menschen diskutieren über Waffen.
Am 1. Oktober 2017 ist ein Angriff in Las Vegas passiert.
Er hat auf andere Menschen geschossen.
Zum Beispiel: Die Gruppe sägt Hochsitze von Jägern um.
Die Gruppe setzt sich für Tiere ein.
Wegen einer Enstcheidung von einem Anker-land, passiert auch immer im Nachbar-land etwas.
Die Bilder sind zusammen mit den Buchstaben.
Anna Faris ist am 29. November 1976 geboren.
Sie ist in der Partei die Grünen.

'''
Here is the output table with my decisions:

| Sentence | Additional Statements | Explanation |
| --- | --- | --- |
| Viele Menschen auf der ganzen Welt haben Angst und sind traurig. | 1 | The prepositional phrase "auf der ganzen Welt" adds one statement, specifying the location. |
| Viele Menschen diskutieren über Waffen. | 1 | The prepositional phrase "über Waffen" adds one statement, specifying the topic of discussion. |
| Am 1. Oktober 2017 ist ein Angriff in Las Vegas passiert. | 2 | The date "Am 1. Oktober 2017" adds one statement, and the prepositional phrase "in Las Vegas" adds another statement, specifying the location. |
| Er hat auf andere Menschen geschossen. | 1 | The prepositional phrase "auf andere Menschen" adds one statement, specifying the target. |
| Zum Beispiel: Die Gruppe sägt Hochsitze von Jägern um. | 1 | The prepositional phrase "von Jägern" adds one statement, specifying the ownership. |
| Die Gruppe setzt sich für Tiere ein. | 1 | The prepositional phrase "für Tiere" adds one statement, specifying the object of support. |
| Wegen einer Enstcheidung von einem Anker-land, passiert auch immer im Nachbar-land etwas. | 2 | The prepositional phrases "von einem Anker-land" and "im Nachbar-land" each add one statement, specifying the source and location of the effect. |
| Die Bilder sind zusammen mit den Buchstaben. | 1 | The prepositional phrase "mit den Buchstaben" adds one statement, specifying the accompaniment. |
| Anna Faris ist am 29. November 1976 geboren. | 1 | The date "am 29. November 1976" adds one statement, specifying the birthdate. The information that she was born is trivial, so it's not an additional statement. |
| Sie ist in der Partei die Grünen. | 1 | The prepositional phrase "in der Partei" adds one statement, specifying the membership. |

Let me know if you have any questions or need further clarification!
'''
