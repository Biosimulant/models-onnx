from __future__ import annotations

from src.imported_bioimage_model import BioimageEmbryonetBaseModel


class _FakeSession:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.last_feed_dict = None

    def get_inputs(self):
        return [
            type("In", (), {"name": "data"})(),
            type("In", (), {"name": "t_for_conv"})(),
            type("In", (), {"name": "t_for_fc"})(),
        ]

    def get_outputs(self):
        return [type("Out", (), {"name": "output"})()]

    def run(self, outputs, feed_dict):
        self.last_feed_dict = feed_dict
        return [[0.1] * 14]


def test_imported_bioimage_model_emits_scores_and_summary() -> None:
    image = type("Sig", (), {"value": [[[[0.0] * 8 for _ in range(8)] for _ in range(3)]],})()
    model = BioimageEmbryonetBaseModel(
        session_factory=_FakeSession,
        image_shape=(1, 3, 8, 8),
        conv_time_shape=(1, 1, 8, 8),
        fc_time_shape=(1, 1),
    )
    model.set_inputs({"image": image})
    model.advance_to(0.1)
    outputs = model.get_outputs()
    assert "development_scores" in outputs
    assert "prediction_summary" in outputs
    assert "t_for_conv" in model._session.last_feed_dict
    assert "t_for_fc" in model._session.last_feed_dict
