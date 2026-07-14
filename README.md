# Fake-news-detection-model

> Fake-news classifier notebook — TF-IDF features + classical ML baseline on a public news corpus.

## Overview
A notebook that ingests a labelled real/fake news dataset, cleans and vectorises the text, trains a classical classifier, and evaluates it on a held-out split. Built as a foundational NLP portfolio piece and the precursor to the multi-model verifier in [Team09_Cybernetics](https://github.com/jaiashwinisatish/Team09_Cybernetics).

## Features
- End-to-end pipeline: text cleaning → TF-IDF vectorization → classifier → evaluation.
- Confusion matrix and classification report for the held-out test set.
- Self-contained — dataset lives in the repo (see caveat below).

## Tech Stack
- **Language:** Python 3 (Jupyter Notebook)
- **Libraries:** scikit-learn, pandas, numpy, nltk

## How to Run
```bash
pip install scikit-learn pandas numpy nltk jupyter
jupyter notebook  # then open the main .ipynb
```

## Caveat
The dataset (~270 MB) is committed to the repo — clone with `--depth 1` if you only want the notebook. A future revision should move the CSV to Git LFS or Kaggle.


## Author
Vijay Nagargoje — [LinkedIn](https://www.linkedin.com/in/vijay-nagargoje) · [GitHub](https://github.com/AI-Vijay)
