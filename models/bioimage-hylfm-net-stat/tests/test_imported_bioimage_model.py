from __future__ import annotations

from src.imported_bioimage_model import BioimageHylfmNetStat


class _FakeSession:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def get_inputs(self):
        return [type("In", (), {"name": "0"})()]

    def get_outputs(self):
        return [type("Out", (), {"name": "220"})()]

    def run(self, outputs, feed_dict):
        return [next(iter(feed_dict.values()))]


def test_imported_bioimage_model_emits_volume_and_summary() -> None:
    signal = type("Sig", (), {"value": [[[[0.0] * 8 for _ in range(8)]]],})()
    model = BioimageHylfmNetStat(session_factory=_FakeSession, input_shape=(1, 1, 8, 8), output_shape=(1, 1, 8, 8))
    model.set_inputs({"lf_image": signal})
    model.advance_to(0.1)
    outputs = model.get_outputs()
    assert "reconstruction_volume" in outputs
    assert "prediction_summary" in outputs
