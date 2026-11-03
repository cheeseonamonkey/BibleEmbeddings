import json

import pytest

from bible_embeddings.corpus import load_passages, overlap
from bible_embeddings.retrieval import Retriever


@pytest.fixture
def source(tmp_path):
    data = [{"name": "Genesis", "chapters": [
        ["In the beginning God created the heaven and the earth.",
         "And the earth was without form, and void.",
         "And God said, Let there be light.",
         "And there was light.",
         "And God saw the light, that it was good.",
         "And God divided the light from the darkness."],
        ["Thus the heavens and the earth were finished."]]}]
    file = tmp_path / "bible.json"
    file.write_text(json.dumps(data))
    return file


def test_windows_do_not_cross_chapters(source):
    passages = load_passages(source)
    assert len(passages) == 6 + 4 + 2 + 1
    assert any(p.reference == "Genesis 1:1–3" for p in passages)
    assert any(p.reference == "Genesis 1:2–6" for p in passages)
    assert not any("2:" in p.reference and "–" in p.reference for p in passages)
    assert all("Genesis 2:" not in p.text for p in passages)


def test_overlap(source):
    passages = load_passages(source, widths=(1,))
    assert overlap(passages[0], passages[0])
    assert not overlap(passages[0], passages[1])
    assert not overlap(passages[0], passages[-1])


def test_retrieval(source):
    passages = load_passages(source)
    r = Retriever(passages)
    matches = r.search("light and darkness", limit=5)
    assert matches
    assert all(not overlap(p, q) for i, (p, _) in enumerate(matches)
               for q, _ in matches[i + 1:])
    assert r.search("light", limit=2) != r.search("void", limit=2)


def test_invalid_input(source):
    with pytest.raises(ValueError):
        load_passages(source, widths=(0,))
    with pytest.raises(ValueError):
        Retriever(load_passages(source)).search("   ")


def test_scoring_deduplicates_within_not_between_stances(source):
    from bible_embeddings.judge import Evidence, top_evidence
    p = load_passages(source, widths=(1,))
    e1 = Evidence(p[0], 0.2, {"entailment": .9, "neutral": .05, "contradiction": .05},
                  {"entailment": .9, "neutral": .05, "contradiction": .05})
    e2 = Evidence(p[0], 0.1, {"entailment": .8, "neutral": .05, "contradiction": .15},
                  {"entailment": .8, "neutral": .05, "contradiction": .15})
    e3 = Evidence(p[1], 0.2, {"entailment": .02, "neutral": .03, "contradiction": .95},
                  {"entailment": .02, "neutral": .03, "contradiction": .95})
    assert top_evidence([e1, e2], "entailment") == [e1]
    assert top_evidence([e1, e3], "contradiction") == [e3]
