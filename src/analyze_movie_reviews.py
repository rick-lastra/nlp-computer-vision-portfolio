"""Explore NLTK's movie-review corpus and compare sentiment vocabularies."""

from collections import Counter
import re
import unicodedata

import matplotlib.pyplot as plt
import nltk
from nltk.corpus import movie_reviews, stopwords


def normalize_tokens(tokens: list[str], stop_words: set[str]) -> list[str]:
    """Lowercase, normalize Unicode, retain alphabetic tokens, and remove stop words."""
    cleaned = []
    for token in tokens:
        normalized = unicodedata.normalize("NFD", token.lower())
        normalized = "".join(char for char in normalized if not unicodedata.combining(char))
        cleaned.extend(
            word for word in re.findall(r"[a-z]+", normalized)
            if len(word) > 1 and word not in stop_words
        )
    return cleaned


def main() -> None:
    nltk.download("movie_reviews", quiet=True)
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))

    counts: dict[str, Counter[str]] = {}
    for sentiment in ("pos", "neg"):
        tokens = movie_reviews.words(categories=[sentiment])
        cleaned = normalize_tokens(tokens, stop_words)
        counts[sentiment] = Counter(cleaned)
        print(
            f"{sentiment}: {len(movie_reviews.fileids(sentiment))} reviews; "
            f"{len(cleaned):,} cleaned tokens; "
            f"{len(counts[sentiment]):,} unique terms"
        )
        print("Most frequent:", counts[sentiment].most_common(15))

    terms = sorted(set(counts["pos"]) | set(counts["neg"]))
    top = sorted(
        terms,
        key=lambda term: counts["pos"][term] + counts["neg"][term],
        reverse=True,
    )[:20]
    positions = range(len(top))
    plt.figure(figsize=(12, 6))
    plt.barh([top[i] for i in positions], [counts["pos"][w] for w in top], label="Positive")
    plt.barh(
        [top[i] for i in positions],
        [counts["neg"][w] for w in top],
        left=[counts["pos"][w] for w in top],
        label="Negative",
    )
    plt.xlabel("Token frequency")
    plt.title("Most frequent cleaned terms in movie reviews")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
