"""Train a spaCy-cleaned TF-IDF Logistic Regression ticket classifier."""

import argparse

import matplotlib.pyplot as plt
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def clean_texts(texts: pd.Series, nlp) -> list[str]:
    cleaned = []
    for doc in nlp.pipe(texts.fillna("").astype(str), batch_size=64):
        cleaned.append(
            " ".join(
                token.lemma_.lower()
                for token in doc
                if not token.is_stop and not token.is_punct and not token.is_space
            )
        )
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Authorized ticket dataset")
    parser.add_argument("--text-column", default="description")
    parser.add_argument("--label-column", default="category")
    args = parser.parse_args()

    data = pd.read_csv(args.csv)
    data = data[[args.text_column, args.label_column]].dropna()
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    texts = clean_texts(data[args.text_column], nlp)
    labels = data[args.label_column].astype(str)

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000)),
            ("classifier", LogisticRegression(max_iter=1_000, class_weight="balanced")),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    print(classification_report(y_test, predictions, zero_division=0))
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, xticks_rotation="vertical")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
