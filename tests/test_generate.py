"""Step 4: basic RAG. Question -> top-k chunks -> prompt -> Claude -> answer.

The first group runs with a fake LLM client (no network). The `live` tests call
the real API and are skipped when no key is configured.
"""
from types import SimpleNamespace

import pytest

from rag.ask import answer
from rag.generate import NOT_COVERED, build_prompt, generate_answer


class FakeClient:
    """Stands in for anthropic.Anthropic(); records the request it was given."""

    def __init__(self, reply="A canned answer."):
        self.reply, self.calls = reply, []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.reply)])


HITS = [
    {"chunk_id": "a:x:00", "text": "The committee shall meet at least four times per year.",
     "metadata": {"title": "Committee Charter", "section": "Administrative Matters"}, "distance": 0.2},
    {"chunk_id": "b:y:00", "text": "We do not accept gifts from suppliers.",
     "metadata": {"title": "Supplier Code", "section": "Interactions"}, "distance": 0.4},
]


def test_prompt_contains_every_chunk_with_its_source_and_the_question():
    system, user = build_prompt("How often does the committee meet?", HITS)
    assert "[1] Committee Charter | Administrative Matters" in user
    assert "[2] Supplier Code | Interactions" in user
    for h in HITS:
        assert h["text"] in user
    assert user.rstrip().endswith("How often does the committee meet?")
    assert user.index("[1]") < user.index("[2]") < user.index("How often")


def test_system_prompt_restricts_the_model_to_the_context():
    system, _ = build_prompt("q", HITS)
    assert "only" in system.lower()
    assert NOT_COVERED in system


def test_generate_sends_the_prompt_and_returns_the_text():
    client = FakeClient("Four times a year.")
    out = generate_answer("How often does the committee meet?", HITS, client=client)
    assert out == "Four times a year."
    (call,) = client.calls
    system, user = build_prompt("How often does the committee meet?", HITS)
    # The API rejects sampling parameters for this model, so none may be sent.
    assert not {"temperature", "top_p", "top_k"} & set(call)
    assert call["system"] == system
    assert call["messages"] == [{"role": "user", "content": user}]
    assert call["model"] and call["max_tokens"] > 0


def test_pipeline_retrieves_top_k_and_passes_them_to_the_llm(collection):
    client = FakeClient()
    result = answer("How many times a year must the ethics committee meet?",
                    k=4, collection=collection, client=client)
    assert result["answer"] == "A canned answer."
    assert len(result["hits"]) == 4
    assert result["hits"][0]["metadata"]["section"] == "Administrative Matters"
    assert result["hits"][0]["text"] in client.calls[0]["messages"][0]["content"]


@pytest.mark.live
def test_live_answer_is_grounded_in_the_retrieved_chunk(collection):
    result = answer("How many times a year must the ethics and compliance committee meet?",
                    collection=collection)
    assert "four" in result["answer"].lower() or "4" in result["answer"]


@pytest.mark.live
def test_live_out_of_scope_question_is_declined(collection):
    result = answer("How many vacation days do employees get per year?", collection=collection)
    assert NOT_COVERED.lower() in result["answer"].lower()
