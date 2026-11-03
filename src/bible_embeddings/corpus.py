"""Load a verse-level corpus and construct contiguous multi-verse passages."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Passage:
    reference: str
    text: str
    context: str
    book: str
    chapter: int
    start: int
    end: int

    @property
    def key(self) -> tuple[str, int, int, int]:
        return self.book, self.chapter, self.start, self.end


def corpus_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_passages(path: str | Path, widths: tuple[int, ...] = (1, 2, 3, 5),
                  context_radius: int = 2) -> list[Passage]:
    """Only contiguous passages within a chapter; no cross-chapter overlap."""
    if context_radius < 0 or not widths or any(w < 1 for w in widths):
        raise ValueError("Invalid window widths or context radius")
    books = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(books, list) or not books:
        raise ValueError("Expected a nonempty list of books")
    output: list[Passage] = []
    for book in books:
        name, chapters = book.get("name"), book.get("chapters")
        if not name or not isinstance(chapters, list):
            raise ValueError("Missing book name or chapters")
        for chapter_no, chapter in enumerate(chapters, 1):
            if not isinstance(chapter, list):
                raise ValueError(f"Invalid chapter {name} {chapter_no}")
            if not all(isinstance(v, str) and v.strip() for v in chapter):
                raise ValueError(f"Empty or invalid verses in {name} {chapter_no}")
            for width in sorted(set(widths)):
                if width > len(chapter):
                    continue
                # Sliding windows: all adjacent verse combinations are searchable.
                # Dense index can later use a faster stride if needed.
                for start in range(0, len(chapter) - width + 1):
                    end = start + width
                    ref = f"{name} {chapter_no}:{start + 1}"
                    if width > 1:
                        ref += f"–{end}"
                    output.append(Passage(
                        ref,
                        " ".join(chapter[start:end]),
                        " ".join(chapter[max(0, start-context_radius):
                                         min(len(chapter), end+context_radius)]),
                        name, chapter_no, start+1, end
                    ))
    return output


def overlap(a: Passage, b: Passage) -> bool:
    return (a.book == b.book and a.chapter == b.chapter
            and a.start <= b.end and b.start <= a.end)
