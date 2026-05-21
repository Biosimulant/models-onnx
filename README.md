# models-onnx

Curated ONNX-first Biosimulant repository for 15 source-derived AI/ML labs. Each lab keeps the bundled ONNX artifact as
the executable source-derived model and exposes conservative Biosimulant ports, runtime evidence, and visualisations.

## What's Inside

- 5 BioImage.io ONNX imports
- 5 MONAI bundle exports to ONNX
- 5 Hugging Face biology, biomedical, or protein-language ONNX exports

All kept labs use:

```text
labs/<slug>/
  lab.yaml
  README.md
  assets/
  models/core/
    model.yaml
    artifacts/model.onnx
    src/
    tests/
  models/visualisation/
    model.yaml
    src/
    tests/
```

## Scientific Accuracy Scope

These labs validate source-faithful ONNX execution and Biosimulant wiring. They do not claim clinical safety, biological
truth, microscopy performance, or paper-figure reproduction unless a source validation dataset or stored reference
comparison is added.

Runtime smoke tests use synthetic tensors when source sample data is unavailable. Synthetic inference proves graph load,
shape compatibility, finite outputs, and visual payloads; it is not scientific evidence for model performance.

## Lab Set

BioImage.io:

- `bioimage-embryonet-embryo-stage-classification`
- `bioimage-hpa-bestfitting-inceptionv3`
- `bioimage-hpa-bestfitting-densenet`
- `bioimage-hylfm-light-field-reconstruction`
- `bioimage-nuclei-segmentation-boundary`

MONAI:

- `monai-spleen-ct-segmentation`
- `monai-pancreas-ct-dints-segmentation`
- `monai-brats-mri-segmentation`
- `monai-spleen-deepedit-annotation`
- `monai-swin-unetr-btcv-segmentation`

Hugging Face:

- `hf-emilyalsentzer-bio-clinicalbert`
- `hf-facebook-esm2-t6-8m-ur50d`
- `hf-dmis-lab-biobert-base-cased-v1-1`
- `hf-facebook-esm2-t12-35m-ur50d`
- `hf-allenai-scibert-scivocab-uncased`

## Validation

Use a Python environment with `biosim`, `onnxruntime==1.22.1`, `numpy`, `pytest`, and `pyyaml`.

```bash
export PYTHONPATH=/Volumes/dem-ssd/imp/projects/Nitoons/Biosimulant/bsim-active/biosim/src
python scripts/validate_manifests.py
python scripts/check_entrypoints.py
python scripts/audit_publish_ready.py
python scripts/validate_onnx_runtime.py
python scripts/audit_readme_assets.py
python -m compileall -q labs scripts tests
python -m pytest -q
```

`scripts/validate_monai_source_parity.py` remains a best-effort parity check for MONAI exports when the source PyTorch
weights are present in the local validation cache.

## License

Dual-licensed: Apache-2.0 for code and CC BY 4.0 for content and inventory metadata, subject to the upstream model
licenses recorded in each lab manifest.
