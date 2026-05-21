"""Validate source-runtime parity for imported MONAI ONNX models.

This script compares source PyTorch bundle models against the imported ONNX
artifacts using the same deterministic synthetic tensor inputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnxruntime as ort
import torch
from monai.networks.nets import DiNTS, DynUNet, SegResNet, SwinUNETR, TopologyInstance, UNet


REPO_ROOT = Path(__file__).resolve().parents[1]
def build_spleen_ct() -> torch.nn.Module:
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="batch",
    )
    state = torch.load(REPO_ROOT / ".validation-cache" / "spleen-ct-model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    return model


def build_brats() -> torch.nn.Module:
    model = SegResNet(
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        init_filters=16,
        in_channels=4,
        out_channels=3,
        dropout_prob=0.2,
    )
    state = torch.load(REPO_ROOT / ".validation-cache" / "brats-model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    return model


def build_deepedit() -> torch.nn.Module:
    model = DynUNet(
        spatial_dims=3,
        in_channels=3,
        out_channels=2,
        kernel_size=[3, 3, 3, 3, 3, 3],
        strides=[1, 2, 2, 2, 2, [2, 2, 1]],
        upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
        norm_name="instance",
        deep_supervision=False,
        res_block=True,
    )
    state = torch.load(REPO_ROOT / ".cache" / "deepedit-model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    return model


def build_swin() -> torch.nn.Module:
    model = SwinUNETR(
        spatial_dims=3,
        in_channels=1,
        out_channels=14,
        feature_size=48,
        use_checkpoint=True,
    )
    state = torch.load(REPO_ROOT / ".cache" / "swin-unetr-model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    return model


def build_pancreas() -> torch.nn.Module:
    arch = torch.load(REPO_ROOT / ".cache" / "pancreas-search-code.pt", map_location="cpu", weights_only=False)
    space = TopologyInstance(
        channel_mul=1,
        num_blocks=12,
        num_depths=4,
        use_downsample=True,
        arch_code=[arch["arch_code_a"], arch["arch_code_c"]],
        device="cpu",
    )
    model = DiNTS(
        dints_space=space,
        in_channels=1,
        num_classes=3,
        use_downsample=True,
        node_a=torch.from_numpy(arch["node_a"]),
    )
    state = torch.load(REPO_ROOT / ".cache" / "pancreas-model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    return model


REGISTRY: dict[str, dict[str, Any]] = {
    "monai-spleen-ct-segmentation": {
        "builder": build_spleen_ct,
        "input_shape": (1, 1, 96, 96, 96),
    },
    "monai-brats-mri-segmentation": {
        "builder": build_brats,
        "input_shape": (1, 4, 128, 128, 128),
    },
    "monai-spleen-deepedit-annotation": {
        "builder": build_deepedit,
        "input_shape": (1, 3, 128, 128, 128),
    },
    "monai-swin-unetr-btcv-segmentation": {
        "builder": build_swin,
        "input_shape": (1, 1, 96, 96, 96),
    },
    "monai-pancreas-ct-dints-segmentation": {
        "builder": build_pancreas,
        "input_shape": (1, 1, 96, 96, 96),
    },
}


def run_one(slug: str, builder: Callable[[], torch.nn.Module], input_shape: tuple[int, ...]) -> dict[str, Any]:
    torch.manual_seed(0)
    np.random.seed(0)
    model = builder()
    model.eval()
    dummy = torch.randn(*input_shape, dtype=torch.float32)

    with torch.no_grad():
        source = model(dummy).detach().cpu().numpy()

    onnx_path = REPO_ROOT / "labs" / slug / "models" / "core" / "artifacts" / "model.onnx"
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = session.run(None, {"image": dummy.numpy()})[0]
    diff = np.abs(source - ort_out)

    return {
        "slug": slug,
        "input_shape": list(input_shape),
        "output_shape": list(ort_out.shape),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
    }


def main() -> None:
    results = []
    for slug, item in REGISTRY.items():
        print(f"Validating {slug}...")
        results.append(run_one(slug, item["builder"], item["input_shape"]))

    out_dir = REPO_ROOT / "validation"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "monai-source-parity.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
