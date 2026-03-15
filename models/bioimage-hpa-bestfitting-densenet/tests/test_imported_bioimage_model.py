from __future__ import annotations

from src.imported_bioimage_model import BioimageHpaBestfittingDensenet


class _FakeSession:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def get_inputs(self):
        return [type("In", (), {"name": "image"})()]

    def get_outputs(self):
        return [type("Out", (), {"name": "classes"})(), type("Out", (), {"name": "features"})()]

    def run(self, outputs, feed_dict):
        return [[[0.1] * 28], [[0.2] * 1024]]


def test_imported_bioimage_model_emits_class_and_feature_outputs() -> None:
    signal = type("Sig", (), {"value": [[[[0.0] * 16 for _ in range(16)] for _ in range(4)]],})()
    model = BioimageHpaBestfittingDensenet(session_factory=_FakeSession, input_shape=(1, 4, 16, 16))
    model.set_inputs({"image": signal})
    model.advance_to(0.1)
    outputs = model.get_outputs()
    assert "class_probabilities" in outputs
    assert "feature_embeddings" in outputs
    assert "prediction_summary" in outputs
