# KlarTextCoders: GermEval 2024 StaGE Submission

Code for the GermEval 2024 "Statement Segmentation in German Easy Language (StaGE)" shared task.

## Citation

```bibtex
@inproceedings{ramarao-etal-2024-klartextcoders,
    title = "{K}lar{T}ext{C}oders at {S}ta{GE}: Automatic Statement Annotations for {G}erman Easy Language",
    author = "Ramarao, Akhilesh Kakolu  and
      Petersen, Wiebke  and
      Stein, Anna Sophia  and
      Stein, Emma  and
      Xia, Hanxin",
    editor = {Schomacker, Thorben  and
      Ansch{\"u}tz, Miriam  and
      Stodden, Regina},
    booktitle = "Proceedings of GermEval 2024 Shared Task on Statement Segmentation in German Easy Language (StaGE)",
    month = sep,
    year = "2024",
    address = "Vienna, Austria",
    publisher = "Association for Computational Lingustics",
    url = "https://aclanthology.org/2024.germeval-1.2/",
    pages = "15--27"
}
```

## Setup

```bash
pip install -r requirements.txt
python -m spacy download de_core_news_sm
python -m spacy download de_dep_news_trf
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
```

For constituency parsing, also install [benepar](https://github.com/nikitakit/self-attentive-parser):

```bash
pip install benepar
python -c "import benepar; benepar.download('benepar_en3'); benepar.download('benepar_de2')"
```

## Data

Place the shared task data in `data/`:
- `train.csv`, `trial.csv`, `test.csv`, `eval.csv` are provided by organizers
- Feature CSVs are written to `data/metrics/`
- Prolog files are written to `data/prologData/`

## Reproducing Results

The scripts are organized to match the methodology sections of the paper (Section 3).

### 3.1 Rule-based (`scripts/rule_based/`)

The rule-based parser counts statements using spaCy dependency parsing and hand-crafted rules based on the annotation guidelines.

```bash
# Compute linguistic features
python scripts/feature_based/basic.py --dataset train
python scripts/feature_based/basic.py --dataset test

# Generate rule-based predictions for statement counts
python scripts/rule_based/rules.py --dataset train
python scripts/rule_based/rules.py --dataset test
```

### 3.2 Feature-based (`scripts/feature_based/`)

#### Feature extraction

**Dependency tree and AMR features:**

```bash
# Step 1: Translate to English, compute AMR, write Prolog input files

python scripts/feature_based/amr_deptree_features.py --action pipeline --split train
python scripts/feature_based/amr_deptree_features.py --action pipeline --split test
python scripts/feature_based/amr_deptree_features.py --action pipeline --split augmented

# Step 2: Run Prolog scripts to extract features

swipl scripts/feature_based/amr.pl    # extracts AMR features
swipl scripts/feature_based/dep_tree.pl  # extracts dependency tree features

# Step 3: Build feature matrices from Prolog output

python scripts/feature_based/amr_deptree_features.py --action features --split train
python scripts/feature_based/amr_deptree_features.py --action features --split test
```

**Constituency parsing features (benepar):**

```bash
python scripts/feature_based/get_benepar_features.py --dataset train
python scripts/feature_based/get_benepar_features.py --dataset test
```

**Additional linguistic features:**

```bash
python scripts/feature_based/amr_deptree_features.py --action additional --split train
python scripts/feature_based/amr_deptree_features.py --action additional --split test
```

**BERT embeddings (CLS token, reduced to 10 dimensions using UMAP):**

```bash
python scripts/feature_based/embeddings.py --dataset train
python scripts/feature_based/embeddings.py --dataset test
```

#### Data augmentation

Round-trip translation via Finnish and Mandarin to balance the training set:

```bash
python scripts/feature_based/amr_deptree_features.py --action augment
```

#### Classification experiments

Run classifiers (Random Forest, SVM, MLP, Logistic Regression) on the extracted features:

```bash
# Full classifier experiments with all features
python scripts/feature_based/amr_deptree_features.py --action classify

# Individual classifier baselines with specific feature sets
python scripts/feature_based/classifier_baselines.py -d combined -c RF
python scripts/feature_based/classifier_baselines.py -d combined -c SVM
python scripts/feature_based/classifier_baselines.py -d combined -c MLP
python scripts/feature_based/classifier_baselines.py -d combined -c regression
```

### 3.3 BERT (`scripts/bert/`)

**Subtask 1: Statement count prediction**

```bash
python scripts/bert/finetune_bert_statement_count.py \
  --data data/train_trial_test.csv \
  --model-name bert-base-german-cased \
  --epochs 50 \
  --batch-size 16
```

**Subtask 2: Statement span prediction (token classification + POS features)**

```bash
python scripts/bert/finetune_bert_with_pos.py \
  --data-path data \
  --stage both \
  --epochs-stage1 8 \
  --epochs-stage2 10
```

Stage 1 trains a BERT model augmented with POS features for statement count prediction. Stage 2 trains a token classifier for statement span prediction.

### 3.4 LLMs (`scripts/llm/`)

We used LLaMA-3-70B-Instruct via the KISSKI API.

1. Request an API key from [KISSKI](https://kisski.gwdg.de/leistungen/2-02-llm-service/) by clicking "Buchen".
2. Log in with your German university credentials via AcademicID.
3. Set your API key in the script:

```python
api_key = "YOUR_API_KEY"
```

4. Run predictions:

```bash
# Statement count prediction
python scripts/llm/api_predictions_num_statements.py

# Statement span prediction
python scripts/llm/api_predictions_statement_spans.py
```

Prompts and example outputs are in `scripts/llm/prompts_subcases.md`. Annotation guidelines used for prompting are in `scripts/llm/annotation_guidelines_cleaned.txt`.

