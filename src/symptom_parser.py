
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import spacy
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer
from spacy.matcher import PhraseMatcher


SymptomState = Literal["PRESENT", "ABSENT", "UNCERTAIN"]


@dataclass(frozen=True)
class SymptomMatch:
    canonical: str
    state: SymptomState
    source_text: str
    method: str
    score: float


class SymptomParser:
    NEGATION_TERMS = {
        "no",
        "not",
        "never",
        "without",
        "neither",
        "nor",
        "dont",
        "don't",
        "doesnt",
        "doesn't",
        "didnt",
        "didn't",
        "havent",
        "haven't",
        "hasnt",
        "hasn't",
    }

    UNCERTAINTY_TERMS = {
        "maybe",
        "perhaps",
        "possibly",
        "might",
        "think",
        "suspect",
        "unsure",
        "uncertain",
    }

    def __init__(
        self,
        model_path: str | Path,
        aliases_path: str | Path,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        fuzzy_threshold: int = 88,
        semantic_threshold: float = 0.62,
    ) -> None:
        bundle = joblib.load(model_path)
        self.symptom_columns: list[str] = list(bundle["symptom_columns"])
        self.symptom_set = set(self.symptom_columns)

        aliases = json.loads(Path(aliases_path).read_text(encoding="utf-8"))

        # Ignore aliases that do not exist in the trained model feature list.
        self.aliases: dict[str, list[str]] = {
            canonical: values
            for canonical, values in aliases.items()
            if canonical in self.symptom_set
        }

        # Guarantee every trained feature has at least a basic alias.
        for canonical in self.symptom_columns:
            self.aliases.setdefault(
                canonical,
                [canonical, canonical.replace("_", " ")],
            )

        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("sentencizer")

        self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        for canonical, values in self.aliases.items():
            patterns = [self.nlp.make_doc(self._normalize(v)) for v in values]
            self.phrase_matcher.add(canonical, patterns)

        self.alias_to_canonical: dict[str, str] = {}
        for canonical, values in self.aliases.items():
            for value in values:
                self.alias_to_canonical[self._normalize(value)] = canonical

        self.alias_texts = list(self.alias_to_canonical.keys())

        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.alias_embeddings = self.embedding_model.encode(
            self.alias_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        self.fuzzy_threshold = fuzzy_threshold
        self.semantic_threshold = semantic_threshold

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower().strip()
        text = text.replace("’", "'")
        text = re.sub(r"[^a-z0-9'\-\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _fragments(text: str) -> list[str]:
        """Break free text into small pieces for fuzzy/semantic matching."""
        parts = re.split(
            r"[,;.!?]|\b(?:and|but|also|with|plus)\b",
            text,
            flags=re.IGNORECASE,
        )
        return [p.strip() for p in parts if len(p.strip()) >= 3]

    def _state_for_span(self, doc, start: int, end: int) -> SymptomState:
        # Small deterministic context window for the demo.
        before_start = max(0, start - 5)
        context_tokens = [
            token.text.lower()
            for token in doc[before_start:start]
        ]

        if any(token in self.NEGATION_TERMS for token in context_tokens):
            return "ABSENT"

        if any(token in self.UNCERTAINTY_TERMS for token in context_tokens):
            return "UNCERTAIN"

        return "PRESENT"

    def _exact_matches(self, normalized_text: str) -> list[SymptomMatch]:
        doc = self.nlp(normalized_text)
        results: list[SymptomMatch] = []

        for match_id, start, end in self.phrase_matcher(doc):
            canonical = self.nlp.vocab.strings[match_id]
            span = doc[start:end]
            state = self._state_for_span(doc, start, end)

            results.append(
                SymptomMatch(
                    canonical=canonical,
                    state=state,
                    source_text=span.text,
                    method="exact_or_synonym",
                    score=1.0,
                )
            )

        return results

    def _fuzzy_match(self, fragment: str) -> SymptomMatch | None:
        match = process.extractOne(
            fragment,
            self.alias_texts,
            scorer=fuzz.WRatio,
        )
        if not match:
            return None

        alias, score, _ = match
        # Prevent short 4-letter generic tokens (like 'pain') from capturing full sentences via substring matching
        if len(alias) <= 4 and len(fragment) >= 10:
            return None

        if score < self.fuzzy_threshold:
            return None

        return SymptomMatch(
            canonical=self.alias_to_canonical[alias],
            state="PRESENT",
            source_text=fragment,
            method="fuzzy",
            score=float(score) / 100.0,
        )

    def _semantic_match(self, fragment: str) -> SymptomMatch | None:
        embedding = self.embedding_model.encode(
            [fragment],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        # With normalized embeddings, dot product == cosine similarity.
        scores = self.alias_embeddings @ embedding
        best_index = int(np.argmax(scores))
        best_score = float(scores[best_index])

        if best_score < self.semantic_threshold:
            return None

        alias = self.alias_texts[best_index]
        return SymptomMatch(
            canonical=self.alias_to_canonical[alias],
            state="PRESENT",
            source_text=fragment,
            method="semantic",
            score=best_score,
        )

    @staticmethod
    def _merge(matches: list[SymptomMatch]) -> list[SymptomMatch]:
        """
        Keep one result per canonical symptom.

        Priority:
        ABSENT > UNCERTAIN > PRESENT for explicit contextual statements,
        then prefer the higher score.
        """
        state_priority = {
            "PRESENT": 1,
            "UNCERTAIN": 2,
            "ABSENT": 3,
        }

        chosen: dict[str, SymptomMatch] = {}
        for item in matches:
            previous = chosen.get(item.canonical)
            if previous is None:
                chosen[item.canonical] = item
                continue

            if state_priority[item.state] > state_priority[previous.state]:
                chosen[item.canonical] = item
            elif (
                state_priority[item.state] == state_priority[previous.state]
                and item.score > previous.score
            ):
                chosen[item.canonical] = item

        # If specific localized symptoms are detected, remove redundant generic "pain"
        specific_localized = {
            "abdominal_pain", "belly_pain", "stomach_pain", "headache",
            "chest_pain", "joint_pain", "knee_pain", "back_pain",
            "bone_pain", "ear_pain", "neck_pain", "hip_joint_pain",
            "muscle_pain"
        }
        if "pain" in chosen and any(s in chosen for s in specific_localized):
            del chosen["pain"]

        return sorted(chosen.values(), key=lambda x: x.canonical)

    def parse(self, text: str) -> list[SymptomMatch]:
        normalized = self._normalize(text)
        if not normalized:
            return []

        print(f"Normalized input: {normalized}")

        exact = self._exact_matches(normalized)
        exact_source_fragments = {
            self._normalize(m.source_text)
            for m in exact
        }

        additional: list[SymptomMatch] = []
        for fragment in self._fragments(normalized):
            if fragment in exact_source_fragments:
                continue

            # Do not use fuzzy/semantic matching for obviously negated clauses
            # in this simple demo implementation. Exact/synonym rules handle
            # explicit negated known symptoms more safely.
            fragment_tokens = set(fragment.split())
            if fragment_tokens & self.NEGATION_TERMS:
                continue

            # 1. Prioritize Semantic Search on natural sentence clauses
            semantic = self._semantic_match(fragment)
            if semantic:
                additional.append(semantic)
                continue

            # 2. Fall back to Fuzzy matching for typos/slight spelling variations
            fuzzy = self._fuzzy_match(fragment)
            if fuzzy:
                additional.append(fuzzy)
                continue

        print(f"Exact matches: {len(exact)}, additional matches: {len(additional)}")

        return self._merge(exact + additional)

    def to_feature_vector(
        self,
        matches: list[SymptomMatch],
    ) -> np.ndarray:
        vector = np.zeros(len(self.symptom_columns), dtype=np.uint8)
        index = {name: i for i, name in enumerate(self.symptom_columns)}

        for match in matches:
            # Only explicitly PRESENT symptoms become 1.
            # ABSENT and UNCERTAIN remain 0 for this binary training dataset.
            if match.state == "PRESENT":
                vector[index[match.canonical]] = 1

        return vector

    def parse_as_dict(self, text: str) -> list[dict]:
        return [asdict(m) for m in self.parse(text)]
