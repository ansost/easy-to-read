### Results from the LLM analysis

**Tldr** :LLMs are not super consistent in their judgement. The best usecase I've found for them is the generation of simplified statements, which I found they do quite well at if prompted correctly.
Predicting number of statements also has okay-ish performance, it gets about 50% accuracy, but I think other approaches will do better here, especially since they are more reproducible.  

But if you're interested in how LLMs handled this, feel free to read on.

I tested out some different things:

1. Asking the model to predict the number of statements
2. Asking the model to predict the statement spans
3. Making the model generate simpler versions of the text (text generation)

I ran [LLAMA 70b instruct](https://huggingface.co/spaces/huggingface/llama-instruct-70B) over [HuggingChat](https://huggingface.co/spaces/huggingface/hugging-chat), basically a GPT-like interface for non-GPT models (can recommend).

I provided the annotation guidelines from the website and used few-shot prompting.

---

### Predicting the amount of statements

So I tried n=30 different sentences and it gets an accuracy of about 50%, which is significantly better than chance, but still not super good. This is an example of its output:

The correct outputs would have been: 1,1,1,3,3,2,2

```
I'll annotate the sentences according to the guidelines. Here are my predictions:

1. Zum Beispiel ein Gerät, \newline dass Musik-noten auf Papier prägen konnte.
Number of statements: 2
Explanation: The sentence contains a main clause "Zum Beispiel ein Gerät" and a subordinate clause "dass Musik-noten auf Papier prägen konnte", which provides additional information about the device.

2. So konnte man die Noten mit den Fingern fühlen.
Number of statements: 1
Explanation: The sentence is a simple sentence with a single main clause, so it contains only one statement.

3. Louis Braille fand die Idee gut.
Number of statements: 1
Explanation: The sentence is a simple sentence with a single main clause, so it contains only one statement.

4. Er hat in der Werkstatt von seinem Vater verschiedene Sachen ausprobiert.
Number of statements: 1
Explanation: The sentence is a simple sentence with a single main clause, so it contains only one statement.

5. Zum Beispiel Dreiecke, Quadrate und Kreise in Leder stanzen.
Number of statements: 2
Explanation: The sentence contains a main clause "Zum Beispiel" and a list of examples "Dreiecke, Quadrate und Kreise in Leder stanzen", which provides additional information. Each item in the list could be considered a separate statement, but according to the guidelines, a sequence of adjectives/adverbials forms a single statement.

6. Als Louis Braille 11 Jahr alt ist, \newline entdeckt er die Nacht-schrift.
Number of statements: 2
Explanation: The sentence contains a main clause "Als Louis Braille 11 Jahr alt ist" and a subordinate clause "entdeckt er die Nacht-schrift", which provides additional information about what happened when Louis Braille was 11 years old.

7. Die Nacht-schrift hat der Hauptmann Charles Barbier erfunden.
Number of statements: 1
Explanation: The sentence is a simple sentence with a single main clause, so it contains only one statement.

Let me know if you have any questions or if you'd like me to explain my reasoning in more detail!
```

### Predicting the statement spans

Performance on this was really bad, it hardly got any right (<10%). This for an LLM might also be harder, because if you get the amount of statements wrong, you'll also get the statement spans wrong. It also didn't really understand that you should not "count" every word in the sentence.

```

    Zum Beispiel ein Gerät, \newline dass Musik-noten auf Papier prägen konnte.
        Number of statements: 2
        Statement spans: [[1, 2, 3, 4], [5, 6, 7, 8, 9]] ( tokens: Zum Beispiel ein Gerät, and dass Musik-noten auf Papier prägen konnte)

    So konnte man die Noten mit den Fingern fühlen.
        Number of statements: 1
        Statement spans: None (only one statement)

    Louis Braille fand die Idee gut.
        Number of statements: 1
        Statement spans: None (only one statement)

    Er hat in der Werkstatt von seinem Vater verschiedene Sachen ausprobiert.
        Number of statements: 1
        Statement spans: None (only one statement)

    Zum Beispiel Dreiecke, Quadrate und Kreise in Leder stanzen.
        Number of statements: 2
        Statement spans: [[1, 2], [3, 4, 5, 6, 7]] (tokens: Zum Beispiel, and Dreiecke, Quadrate und Kreise in Leder stanzen)

    Als Louis Braille 11 Jahr alt ist, \newline entdeckt er die Nacht-schrift.
        Number of statements: 2
        Statement spans: [[1, 2, 3, 4, 5], [6, 7, 8, 9]] (tokens: Als Louis Braille 11 Jahr alt ist, and entdeckt er die Nacht-schrift)

    Die Nacht-schrift hat der Hauptmann Charles Barbier erfunden.
        Number of statements: 1
        Statement spans: None (only one statement)

Please let me know if you'd like me to clarify or correct any of these predictions!
```

### Generating simplified versions of the text

The regenerated texts were actually not bad and really good in terms of simplification. You do have
to remind the model to make the sentences into grammatically correct sentences to get good output though.

```

    Zum Beispiel ein Gerät, \newline dass Musik-noten auf Papier prägen konnte.
        Simplified: Ein Gerät prägt Musik-noten auf Papier. (1 statement)
    So konnte man die Noten mit den Fingern fühlen.
        Simplified: Man fühlt die Noten mit den Fingern. (1 statement)
    Louis Braille fand die Idee gut.
        Simplified: Louis Braille findet die Idee gut. (1 statement)
    Er hat in der Werkstatt von seinem Vater verschiedene Sachen ausprobiert.
        Simplified: Er probiert verschiedene Sachen aus. (1 statement)
    Zum Beispiel Dreiecke, Quadrate und Kreise in Leder stanzen.
        Simplified: Er stanzt Dreiecke, Quadrate und Kreise in Leder. (1 statement)
    Als Louis Braille 11 Jahr alt ist, \newline entdeckt er die Nacht-schrift.
        Simplified: Louis Braille entdeckt die Nacht-schrift. (1 statement)
    Die Nacht-schrift hat der Hauptmann Charles Barbier erfunden.
        Simplified: Charles Barbier erfand die Nacht-schrift. (1 statement)
```
