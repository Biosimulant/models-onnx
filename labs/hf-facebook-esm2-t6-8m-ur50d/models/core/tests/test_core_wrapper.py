from __future__ import annotations

import importlib
import sys
from pathlib import Path

import yaml


CORE = Path(__file__).resolve().parents[1]
BSIM_SRC = Path("/Volumes/dem-ssd/imp/projects/Nitoons/Biosimulant/bsim-active/biosim/src")
if str(BSIM_SRC) not in sys.path:
    sys.path.insert(0, str(BSIM_SRC))
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))


def _load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


class _FakeValue:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSession:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        manifest = _load_yaml(CORE / "model.yaml")
        onnx = manifest.get("onnx", {})
        self._inputs = [_FakeValue(str(item["name"])) for item in onnx.get("inputs", [])]
        self._outputs = [_FakeValue(str(item["name"])) for item in onnx.get("outputs", [])]

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, outputs, feed_dict):
        result = []
        for index, _name in enumerate(outputs):
            if index == 0:
                result.append([[0.1, 0.2, 0.3]])
            else:
                result.append([[0.4, 0.5, 0.6]])
        return result


class _Signal:
    def __init__(self, value):
        self.value = value


def _entrypoint():
    manifest = _load_yaml(CORE / "model.yaml")
    module_name, attr = manifest["biosim"]["entrypoint"].split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr), manifest


def test_core_wrapper_emits_declared_outputs_with_fake_session() -> None:
    cls, manifest = _entrypoint()
    kwargs = dict(manifest["biosim"].get("init_kwargs", {}))
    kwargs["session_factory"] = _FakeSession
    for key in ("input_shape", "image_shape", "conv_time_shape", "fc_time_shape", "output_shape"):
        if key in kwargs:
            kwargs[key] = [1] if key != "fc_time_shape" else [1, 1]
    module = cls(**kwargs)
    signals = {}
    for name in module.inputs():
        signals[name] = _Signal([1])
    module.set_inputs(signals)
    module.advance_window(0.0, 0.1)
    outputs = module.get_outputs()
    assert set(manifest["io"]["outputs"]) <= set(outputs)
    assert all(signal.value is not None for signal in outputs.values())
