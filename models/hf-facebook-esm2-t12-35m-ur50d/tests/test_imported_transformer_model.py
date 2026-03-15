from __future__ import annotations

from src.imported_transformer_model import HfFacebookEsm2T1235mUr50d


class _FakeSession:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def get_inputs(self):
        return [type("In", (), {"name": "input_ids"})(), type("In", (), {"name": "attention_mask"})()]

    def get_outputs(self):
        return [type("Out", (), {"name": "last_hidden_state"})(), type("Out", (), {"name": "pooled_output"})()]

    def run(self, outputs, feed_dict):
        batch = len(next(iter(feed_dict.values())))
        seq_length = len(feed_dict["input_ids"][0])
        sequence = [[[float(i)] * 480 for i in range(seq_length)] for _ in range(batch)]
        pooled = [[sum(feed_dict["attention_mask"][0])] * 480]
        return [sequence, pooled]


def test_imported_transformer_model_emits_sequence_and_pooled_outputs() -> None:
    tokens = type("Sig", (), {"value": [1, 2, 3, 4]})()
    mask = type("Sig", (), {"value": [1, 1, 1, 1]})()
    model = HfFacebookEsm2T1235mUr50d(session_factory=_FakeSession, seq_length=4)
    model.set_inputs({"token_ids": tokens, "attention_mask": mask})
    model.advance_to(0.1)
    outputs = model.get_outputs()
    assert "sequence_embeddings" in outputs
    assert "pooled_embedding" in outputs
