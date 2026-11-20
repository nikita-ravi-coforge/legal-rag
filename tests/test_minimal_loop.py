"""Step 2: the smallest embed -> store -> retrieve loop, on two known texts."""
from rag.minimal_loop import TEXTS, build_collection, embed, query


def test_embedding_is_a_1024_dim_vector():
    vectors = embed(["hello"])
    assert len(vectors) == 1 and len(vectors[0]) == 1024


def test_both_texts_are_stored():
    assert build_collection().count() == 2 == len(TEXTS)


def test_gift_question_retrieves_the_gift_text_first():
    hits = query(build_collection(), "Can I accept a present from a vendor?")
    assert [h["id"] for h in hits] == ["gifts", "meetings"]
    assert hits[0]["distance"] < hits[1]["distance"]


def test_meeting_question_retrieves_the_meeting_text_first():
    hits = query(build_collection(), "How often does the committee meet?")
    assert hits[0]["id"] == "meetings"
