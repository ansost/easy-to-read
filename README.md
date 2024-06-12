# Code for the GermEval-task 2024

```
09.03.2024 - Trial data ready
14.04.2024 - Train data ready
18.05.2024 - Test data ready
13.06.2024 - Evaluation start
25.06.2024 - Evaluation end
01.07.2024 - Paper submission due
20.07.2024 - Camera ready due
13.09.2024 - Workshop date
```

Resources:

- [Task Website](https://german-easy-to-read.github.io/statements/)
- [Data](https://github.com/german-easy-to-read/statements/tree/master/data)

# Docs

## Download necessary data

```python
import benepar
import nltk

benepar.download('benepar_de2')
```

```sh
python -m spacy download de_core_news_sm
python -m spacy download de_dep_news_trf
```

## Reproduce classifier results
>
>Download the new train data!

Use the shell script (`scripts/classifier_baselines.sh`) or the command line. For the shell script:

```bash
sh run_classifier_baselines.sh > ../results/classifier_baselines.txt
```

## Testing

```
pytest
```
