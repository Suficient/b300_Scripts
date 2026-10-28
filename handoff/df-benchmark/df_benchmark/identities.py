"""The identity pool: Clipfarmer's person index. The only module that knows where people come from."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np


class Pool:
    def __init__(self, db: Path, root: Path):
        self.conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        self.root = root

    def nearest(self, emb: np.ndarray, same_person_sim: float, exclude: set[int] = frozenset()) -> tuple[int, float] | None:
        """The most similar person as (person_id, cosine), skipping anyone similar enough to be the same person,
        anyone in exclude and anyone without a usable clip."""
        rows = self.conn.execute(
            "SELECT id, emb_sum FROM persons WHERE id IN"
            " (SELECT person_id FROM clips WHERE fate='accepted' AND path != '')").fetchall()
        rows = [r for r in rows if r[0] not in exclude]
        if not rows:
            return None
        m = np.stack([np.frombuffer(b, np.float32) for _, b in rows])
        m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
        sims = m @ emb.astype(np.float32)
        for i in np.argsort(-sims):
            if sims[i] < same_person_sim:
                return rows[i][0], float(sims[i])
        return None

    def person_of(self, clip: Path) -> int | None:
        """The person id Clipfarmer gave this clip (matched by file name), if it has one."""
        suffix = "/" + clip.name  # not LIKE: "_" in clip names would match any character
        row = self.conn.execute("SELECT person_id FROM clips WHERE path = ? OR substr(path, -length(?)) = ?"
                                " ORDER BY id LIMIT 1", (clip.name, suffix, suffix)).fetchone()
        return row[0] if row else None

    def best_clip(self, person_id: int) -> tuple[Path, float | None]:
        """This person's accepted clip with the highest window score (Clipfarmer's results_json "score"),
        and that score. Clips without a score come last; ties go to the earliest clip."""
        path, score = self.conn.execute(
            "SELECT path, json_extract(results_json, '$.score') AS s FROM clips"
            " WHERE person_id=? AND fate='accepted' AND path != ''"
            " ORDER BY s IS NULL, s DESC, id LIMIT 1", (person_id,)).fetchone()
        return self.root / path, score
