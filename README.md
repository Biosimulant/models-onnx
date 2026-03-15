# models-onnx

Curated ONNX-first repository for a 15-model published AI/ML shortlist that
BioSimulant can wrap behind the standard `biosim.BioModule` contract.

## What's Inside

Current shortlist counts:

- `5` BioImage models
- `5` MONAI bundles
- `5` Hugging Face biology / biomedical models

Current readiness split:

- `5` `standardized-near-ready`
- `5` `convertible-needs-export`
- `5` `likely-convertible-transformer`

Current implementation status:

- `15` imported
- `0` scaffolded

## First Published BioImage Batch

The first published BioImage ONNX batch has been materialized into real indexed
model folders:

- `bioimage-nucleisegmentationboundarymodel`
- `bioimage-hpa-bestfitting-inceptionv3`
- `bioimage-hpa-bestfitting-densenet`
- `bioimage-embryonet-base-model`
- `bioimage-hylfm-net-stat`

Those folders include:

- `model.yaml`
- `src/` BioModule wrappers
- `tests/`
- `import-metadata.json`
- `artifacts/model.onnx`
- `artifacts/source-url.txt`

All five of those BioImage models are now materialized and indexed.

## MONAI Batch

The MONAI export batch is now materialized and indexed as real imported model folders:

- `monai-spleen-ct-segmentation`
- `monai-pancreas-ct-dints-segmentation`
- `monai-brats-mri-segmentation`
- `monai-spleen-deepedit-annotation`
- `monai-swin-unetr-btcv-segmentation`

These were published as MONAI bundles rather than ready-made ONNX artifacts, so
each one was exported into a checked-in ONNX model and validated locally. Each
folder includes:

- `model.yaml`
- `src/` wrapper
- `tests/`
- `import-metadata.json`
- `artifacts/model.onnx`
- `artifacts/export-metadata.json`

All five MONAI models are now imported and indexed.

## Transformer Batch

The lighter transformer export batch is now materialized and indexed as real
imported model folders:

- `hf-emilyalsentzer-bio-clinicalbert`
- `hf-facebook-esm2-t6-8m-ur50d`
- `hf-dmis-lab-biobert-base-cased-v1-1`
- `hf-facebook-esm2-t12-35m-ur50d`
- `hf-allenai-scibert-scivocab-uncased`

Each folder includes:

- `model.yaml`
- `src/` wrapper
- `tests/`
- `import-metadata.json`
- `artifacts/model.onnx`
- `artifacts/export-metadata.json`

All five transformer models are now imported and indexed.

## First-Wave Scope

This repository is the current 15-model implementation pass for:

- BioImage imports with strong existing metadata
- MONAI bundles that can be exported or adapted into ONNX
- simpler transformer checkpoints that are realistic ONNX export targets

This repository is the active short list for what it carries right now.

## Validation Status

Operational import and runtime validation are complete for the active 15-model
set. Project-side scientific validation planning and tracking are maintained
outside the published repository.

## Layout

```text
models-onnx/
├── biosim-index.yaml
├── models/
│   ├── bioimage-*/
│   ├── monai-*/
│   └── hf-*/
├── scripts/
├── README.md
└── STANDARDS.md
```

## Getting Started

Imported models depend on `biosim` plus ONNX Runtime:

```bash
pip install "biosim @ git+https://github.com/BioSimulant/biosim.git@main"
pip install onnxruntime==1.22.1
```

## License

Dual-licensed: Apache-2.0 (code), CC BY 4.0 (content and inventory metadata).
