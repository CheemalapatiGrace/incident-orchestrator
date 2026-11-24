"""Small from-scratch BM25 retrieval over the bundled runbooks (no external deps)."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

RUNBOOK_DIR = Path(__file__).parent / "runbooks"
_TOKEN = re.compile(r"[a-z0-9_]+")


@dataclass
class RunbookDoc:
    doc_id: str
    title: str
    category: str
    tags: list[str]
    body: str
    steps: list[str] = field(default_factory=list)


def parse_runbook(text: str) -> RunbookDoc:
    meta: dict[str, str] = {}
    body = text
    if text.startswith("---"):
        _, fm, body = text.split("---", 2)
        for line in fm.strip().splitlines():
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    steps: list[str] = []
    in_steps = False
    for line in body.splitlines():
        if line.startswith("## "):
            in_steps = line.lower().startswith("## steps")
        elif in_steps and re.match(r"\s*\d+\.", line):
            steps.append(re.sub(r"^\s*\d+\.\s*", "", line).strip())
    return RunbookDoc(meta.get("id", "unknown"), meta.get("title", ""), meta.get("category", "unknown"), meta.get("tags", "").split(), body, steps)


def tokenize(s: str) -> list[str]:
    return _TOKEN.findall(s.lower())


class RunbookIndex:
    """Okapi BM25 with title/tag field boosting."""

    def __init__(self, docs: list[RunbookDoc], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = docs
        self.k1, self.b = k1, b
        self._tf: list[Counter[str]] = []
        for d in docs:
            toks = tokenize(d.body) + tokenize(d.title) * 3 + [t for tag in d.tags for t in tokenize(tag)] * 2
            self._tf.append(Counter(toks))
        self._len = [sum(c.values()) for c in self._tf]
        self._avg = sum(self._len) / max(1, len(docs))
        df: Counter[str] = Counter()
        for c in self._tf:
            df.update(c.keys())
        n = len(docs)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    @classmethod
    def load_bundled(cls, extra: dict[str, str] | None = None) -> RunbookIndex:
        docs = [parse_runbook(p.read_text()) for p in sorted(RUNBOOK_DIR.glob("*.md"))]
        if extra:  # used by the eval suite to poison a runbook with injected text
            for d in docs:
                if d.doc_id in extra:
                    d.body += "\n" + extra[d.doc_id]
                    d.steps.append(extra[d.doc_id])
        return cls(docs)

    def search(self, query: str, top_k: int = 3) -> list[tuple[RunbookDoc, float]]:
        q = tokenize(query)
        scored: list[tuple[RunbookDoc, float]] = []
        for doc, tf, ln in zip(self.docs, self._tf, self._len, strict=True):
            s = 0.0
            for term in set(q):
                f = tf.get(term, 0)
                if f:
                    s += self._idf[term] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self._avg))
            if s > 0:
                scored.append((doc, s))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]
