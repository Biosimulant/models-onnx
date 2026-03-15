from __future__ import annotations

from src.imported_monai_model import MonaiBratsMriSegmentation


class _FakeSession:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def get_inputs(self):
        return [type("In", (), {"name": "image"})()]

    def get_outputs(self):
        return [type("Out", (), {"name": "pred"})()]

    def run(self, outputs, feed_dict):
        return [next(iter(feed_dict.values()))]


def test_imported_monai_model_emits_prediction_and_summary() -> None:
    signal = type("Sig", (), {"value": [[[[[0.0] * 8 for _ in range(8)] for _ in range(8)] for _ in range(4)]],})()
    model = MonaiBratsMriSegmentation(session_factory=_FakeSession, input_shape=(1, 4, 8, 8, 8), output_shape=(1, 4, 8, 8, 8))
    model.set_inputs({"volume_tensor": signal})
    model.advance_to(0.1)
    outputs = model.get_outputs()
    assert "class_probabilities" in outputs
    assert "segmentation_summary" in outputs
