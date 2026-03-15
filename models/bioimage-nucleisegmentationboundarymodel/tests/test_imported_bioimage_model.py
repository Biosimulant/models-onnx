from __future__ import annotations

from src.imported_bioimage_model import BioimageNucleisegmentationboundarymodel


class _FakeSession:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def get_inputs(self):
        return [type("In", (), {"name": "input.1"})()]

    def get_outputs(self):
        return [type("Out", (), {"name": "506"})()]

    def run(self, outputs, feed_dict):
        return [next(iter(feed_dict.values()))]


def test_imported_bioimage_model_emits_prediction_and_summary() -> None:
    signal = type("Sig", (), {"value": [[[[0.0] * 256 for _ in range(256)]]],})()
    model = BioimageNucleisegmentationboundarymodel(session_factory=_FakeSession)
    model.set_inputs({"image": signal})
    model.advance_to(0.1)
    outputs = model.get_outputs()
    assert "segmentation_logits" in outputs
    assert "prediction_summary" in outputs
