# Model & Import Standards

Standards for every selected model and import record in `models-onnx`.

## Repo Scope

`models-onnx` is for AI/ML models that are either:

- already distributed as ONNX artifacts
- realistically exportable to ONNX
- worth wrapping inside the BioSimulant model contract

Out of scope for this repository:

- mechanistic standards like SBML / CellML / NeuroML
- extremely heavy external systems that need bespoke infrastructure before they
  can be treated as normal ONNX models

## Directory Layout

Every selected model folder must follow:

```text
models/<family>-<subject>-<role>/
├── model.yaml
├── artifacts/
│   └── <artifact>.onnx
├── src/
│   └── <module>.py
└── tests/
    └── test_<module>.py
```

## Manifest Rules

All models must declare:

```yaml
schema_version: "2.0"
title: "<Domain>: <ModelName>"
description: "<Short description>"
standard: onnx
tags: [onnx, ...]
authors: ["Biosimulant Team"]
biosim:
  entrypoint: "src.<module>:<ClassName>"
runtime:
  dependencies:
    packages:
      - onnxruntime==1.22.1
onnx:
  task: <task>
  model_file: artifacts/<artifact>.onnx
```

Additional rules:

- `onnx.model_file` must point to a checked-in artifact for implemented models.
- `io.inputs` and `io.outputs` must match the BioModule-facing ports.
- Every ONNX model must be wrapped by a normal Python `BioModule`.
- Imported third-party model families must document preprocessing assumptions in
  code and tests, not only in README text.

## Planning Metadata

Project-side planning inventory and validation trackers are maintained outside
the published repository. The published repo should only keep runnable model
assets, manifests, wrappers, tests, and essential repo documentation.

## Import Families

Current first-wave implementation families:

- `bioimage-onnx-image-model`
- `monai-onnx-image-model`
- `transformer-onnx-sequence-encoder`

These families describe how a source model should be wrapped in BioSimulant. A
candidate does not become an implemented model until it has a checked-in
artifact, manifest, wrapper, and tests.
