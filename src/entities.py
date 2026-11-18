from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression

from .utils import clean_text

LOGGER = logging.getLogger(__name__)


def get_taxonomy_columns(taxonomy: pd.DataFrame) -> list[str]:
    columns = [c for c in taxonomy.columns if c.lower().startswith("category_l")]
    return sorted(columns, key=lambda c: int(c.lower().replace("category_l", "")))


def get_taxonomy_paths(taxonomy: pd.DataFrame) -> list[tuple[str, ...]]:
    cols = get_taxonomy_columns(taxonomy)
    paths: list[tuple[str, ...]] = []
    for _, row in taxonomy.iterrows():
        path = tuple(str(row.get(c, "")).strip() for c in cols)
        path = tuple(value for value in path if value and value.lower() not in {"nan", "none"})
        if path and path not in paths:
            paths.append(path)
    if not paths:
        raise ValueError("Taxonomy contains no non-empty category paths.")
    return paths


def compose_erp_text(row: pd.Series) -> str:
    fields = [
        row.get("material_short_text", ""), row.get("service_short_text", ""),
        row.get("material_type", ""), row.get("spend_type", ""),
        row.get("opex_capex", ""), row.get("site_name", ""),
    ]
    return " ".join(str(v) for v in fields if clean_text(v))


def compose_invoice_text(row: pd.Series) -> str:
    fields = [row.get("line_text", ""), row.get("supplier_name_as_captured", "")]
    return " ".join(str(v) for v in fields if clean_text(v))


class ZeroShotTaxonomyClassifier:
    """Taxonomy-only classifier using optional sentence embeddings or offline TF-IDF."""

    def __init__(
        self,
        taxonomy: pd.DataFrame,
        category_keywords: dict[str, list[str]] | None = None,
        embedding_model: str | None = None,
    ):
        self.paths = get_taxonomy_paths(taxonomy)
        category_keywords = category_keywords or {}
        self.documents = []
        for path in self.paths:
            synonyms = category_keywords.get(path[-1], [])
            self.documents.append(clean_text(" ".join((*path, *synonyms))))

        self.embedding_backend = None
        self.embedding_model = None
        self.taxonomy_embeddings = None
        if embedding_model:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer(embedding_model)
                self.taxonomy_embeddings = self.embedding_model.encode(
                    self.documents, normalize_embeddings=True, show_progress_bar=False
                )
                self.embedding_backend = "sentence_transformers"
                LOGGER.info("Using sentence-transformer taxonomy classifier: %s", embedding_model)
            except Exception as exc:
                LOGGER.warning("Could not load embedding model '%s' (%s); falling back to TF-IDF.", embedding_model, exc)

        if self.embedding_backend is None:
            self.embedding_backend = "tfidf"

        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), min_df=1,
            sublinear_tf=True, max_features=120_000,
        )
        self.word_vectorizer = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=1,
            sublinear_tf=True, max_features=80_000, token_pattern=r"(?u)\b\w+\b",
        )
        self.char_taxonomy = self.char_vectorizer.fit_transform(self.documents)
        self.word_taxonomy = self.word_vectorizer.fit_transform(self.documents)

    def predict(self, texts: list[str], batch_size: int = 1000) -> list[tuple[tuple[str, ...], float]]:
        results: list[tuple[tuple[str, ...], float]] = []
        for start in range(0, len(texts), batch_size):
            batch = [clean_text(t) for t in texts[start:start + batch_size]]
            if self.embedding_backend == "sentence_transformers" and self.embedding_model is not None:
                query_embeddings = self.embedding_model.encode(
                    batch, normalize_embeddings=True, show_progress_bar=False,
                    batch_size=min(batch_size, 64),
                )
                scores = np.asarray(query_embeddings) @ np.asarray(self.taxonomy_embeddings).T
                for idx, text in enumerate(batch):
                    if not text:
                        results.append((self.paths[0], 0.0))
                    else:
                        best = int(np.argmax(scores[idx]))
                        results.append((self.paths[best], float(np.clip(scores[idx, best], 0.0, 1.0))))
                continue
            char = self.char_vectorizer.transform(batch)
            word = self.word_vectorizer.transform(batch)
            scores = 0.60 * cosine_similarity(char, self.char_taxonomy) + 0.40 * cosine_similarity(word, self.word_taxonomy)
            for idx, text in enumerate(batch):
                if not text or scores.shape[1] == 0:
                    results.append((self.paths[0], 0.0))
                else:
                    best = int(np.argmax(scores[idx]))
                    results.append((self.paths[best], float(np.clip(scores[idx, best], 0.0, 1.0))))
        return results


class SupervisedTaxonomyClassifier:
    """Supervised text classifier trained only on the supplied gold-labelled ERP rows."""

    def __init__(self, taxonomy: pd.DataFrame, random_state: int = 42):
        self.paths = get_taxonomy_paths(taxonomy)
        self.path_to_class = {path: i for i, path in enumerate(self.paths)}
        self.class_to_path = {i: path for path, i in self.path_to_class.items()}
        self.vectorizer = FeatureUnion([
            ("word", TfidfVectorizer(
                analyzer="word", ngram_range=(1, 2), min_df=2,
                max_features=120_000, sublinear_tf=True,
                token_pattern=r"(?u)\b\w+\b", strip_accents="unicode",
            )),
            ("char", TfidfVectorizer(
                analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                max_features=140_000, sublinear_tf=True, strip_accents="unicode",
            )),
        ])
        self.model = LogisticRegression(
            max_iter=250, C=4.0, class_weight="balanced",
            solver="lbfgs", random_state=random_state,
        )
        self.is_fitted = False

    def fit(self, texts: list[str], paths: list[tuple[str, ...]]) -> "SupervisedTaxonomyClassifier":
        valid_indices = [i for i, path in enumerate(paths) if path in self.path_to_class and clean_text(texts[i])]
        if len(valid_indices) < 10:
            raise ValueError(f"Only {len(valid_indices)} usable labelled rows; need at least 10 to train.")
        x = [texts[i] for i in valid_indices]
        y = [self.path_to_class[paths[i]] for i in valid_indices]
        x_matrix = self.vectorizer.fit_transform(x)
        self.model.fit(x_matrix, y)
        self.is_fitted = True
        return self

    def predict(self, texts: list[str], batch_size: int = 2000) -> list[tuple[tuple[str, ...], float]]:
        if not self.is_fitted:
            raise RuntimeError("The supervised classifier has not been fitted.")
        results = []
        for start in range(0, len(texts), batch_size):
            matrix = self.vectorizer.transform(texts[start:start + batch_size])
            probabilities = self.model.predict_proba(matrix)
            class_ids = self.model.classes_
            for row in probabilities:
                local = int(np.argmax(row))
                class_id = int(class_ids[local])
                results.append((self.class_to_path[class_id], float(row[local])))
        return results


def evaluate_supervised_holdout(
    taxonomy: pd.DataFrame,
    texts: list[str],
    target_paths: list[tuple[str, ...]],
    test_size: float = 0.20,
    random_state: int = 42,
) -> dict[str, Any]:
    valid = [i for i, p in enumerate(target_paths) if p in get_taxonomy_paths(taxonomy) and clean_text(texts[i])]
    y = [" > ".join(target_paths[i]) for i in valid]
    if len(valid) < 20:
        return {"status": "skipped", "reason": "Fewer than 20 usable labelled rows."}
    counts = pd.Series(y).value_counts()
    stratify = y if counts.min() >= 2 else None
    train_idx, test_idx = train_test_split(
        valid, test_size=test_size, random_state=random_state,
        stratify=stratify,
    )
    model = SupervisedTaxonomyClassifier(taxonomy, random_state=random_state)
    model.fit([texts[i] for i in train_idx], [target_paths[i] for i in train_idx])
    predictions = model.predict([texts[i] for i in test_idx])
    pred = [" > ".join(path) for path, _ in predictions]
    truth = [" > ".join(target_paths[i]) for i in test_idx]
    return {
        "status": "ok",
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "top1_accuracy": round(float(accuracy_score(truth, pred)), 4),
        "macro_f1": round(float(f1_score(truth, pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(truth, pred, average="weighted", zero_division=0)), 4),
        "note": "Held-out split; paths are compared at the complete leaf path level.",
    }


def evaluate_zero_shot(
    predictions: list[tuple[tuple[str, ...], float]],
    target_paths: list[tuple[str, ...]],
) -> dict[str, Any]:
    valid = [(predictions[i][0], target_paths[i]) for i in range(min(len(predictions), len(target_paths))) if target_paths[i]]
    if not valid:
        return {"status": "skipped", "reason": "No aligned labels to evaluate."}
    full = sum(pred == truth for pred, truth in valid) / len(valid)
    level_scores: dict[str, float] = {}
    max_level = max(max(len(p), len(t)) for p, t in valid)
    for level in range(max_level):
        comparable = [(p[level], t[level]) for p, t in valid if len(p) > level and len(t) > level]
        if comparable:
            level_scores[f"l{level + 1}_accuracy"] = round(
                sum(p == t for p, t in comparable) / len(comparable), 4
            )
    return {
        "status": "ok",
        "rows_evaluated": len(valid),
        "complete_path_accuracy": round(full, 4),
        **level_scores,
        "note": "Predictions used taxonomy text only; labels were consulted afterwards for evaluation.",
    }


def path_from_label_row(row: pd.Series, level_columns: list[str]) -> tuple[str, ...]:
    values = []
    for col in level_columns:
        value = str(row.get(col, "")).strip()
        if value and value.lower() not in {"nan", "none"}:
            values.append(value)
        else:
            break
    return tuple(values)
