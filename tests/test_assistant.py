"""The assistant and recommender together, without calling Groq.

These would have caught the explain_it mismatch: assistant.chat calls recommend(),
so any change to recommend()'s signature now breaks a test instead of the live chat.
"""
import server.assistant as assistant
from server.recommender import recommend

GARBAGE = {"name": "Garbage (clean)", "train_count": 2275, "num_classes": 6}
CIFAR = {"name": "CIFAR-10", "train_count": 50_000, "num_classes": 10}


def test_recommend_rules():
    assert recommend(CIFAR, ["cpu", "cpu"], "accurate", explain_it=False)["model"] == "resnet20"
    assert recommend(CIFAR, ["cpu", "cpu"], "fastest", explain_it=False)["model"] == "cnn_small"
    small = recommend(GARBAGE, ["cpu", "cpu"], "accurate", explain_it=False)
    assert small["model"] == "cnn_bn" and small["small_dataset"]


def test_chat_without_key_falls_back_to_rules(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    r = assistant.chat([{"role": "user", "content": "Which model?"}], GARBAGE, ["cpu"], "accurate", 10, 1)
    assert r["suggested_model"] == "cnn_bn"
    assert "unavailable" in r["reply"]


class FakeResponse:
    status_code = 200

    def __init__(self, text):
        self.text = text

    def json(self):
        return {"choices": [{"message": {"content": self.text}}]}


def test_chat_suggestion_is_parsed_and_checked(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    assistant._recent_calls.clear()
    replies = iter(["Use the **batch-norm CNN**: it suits small data. SUGGEST: cnn_bn",
                    "Try this one. SUGGEST: imaginary_model"])
    monkeypatch.setattr(assistant.requests, "post", lambda *a, **k: FakeResponse(next(replies)))

    r = assistant.chat([{"role": "user", "content": "Which?"}], GARBAGE, ["cpu"], "accurate", 10, 1)
    assert r["suggested_model"] == "cnn_bn"
    assert "SUGGEST" not in r["reply"] and "**" not in r["reply"]

    r = assistant.chat([{"role": "user", "content": "Which?"}], GARBAGE, ["cpu"], "accurate", 10, 1)
    assert r["suggested_model"] is None                     # a model that doesn't exist is ignored
    assert "SUGGEST" not in r["reply"]