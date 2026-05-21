# Bioimage Nucleisegmentationboundarymodel

This Biosimulant lab wraps a source-derived ONNX artifact and reports runtime evidence from the bundled graph.

## Scientific Context

Nuclei boundary segmentation from a BioImage.io ONNX graph.

## Source And Artifact

- `rdf`: https://bioimage-io.github.io/collection-bioimage-io/rdfs/10.5281/zenodo.5764892/6647674/rdf.yaml
- `onnx`: https://zenodo.org/api/records/6647674/files/weights.onnx/content

- ONNX artifact: `models/core/artifacts/model.onnx`
- Runtime: ONNX Runtime CPU execution provider
- Validation scope: graph load, input/output contract, finite synthetic inference, Biosimulant wiring, and visual payloads

## Inputs

- `nuclei_image`: Input maps to ONNX graph input `input.1` and is an already-preprocessed nuclei image tensor.

The lab does not add raw image, raw text, raw protein sequence, tokenizer, or medical-image preprocessing unless that
preprocessing is bundled and validated. Public inputs are already-preprocessed tensors matching the ONNX graph contract.

## Outputs

- `segmentation_logits`: Compact tensor evidence derived from the source-derived ONNX graph output.
- `prediction_summary`: Compact summary derived from the ONNX graph output tensor.

## Visualisations

The visualisation answers:

> What segmentation-logit tensor does this source nuclei-boundary ONNX graph emit for the supplied preprocessed nuclei image?

It shows provenance, graph schema, runtime tensor evidence, and a conservative caveat. Synthetic inference is a runtime
smoke test, not biological, clinical, or microscopy performance evidence.

<!-- BIOSIMULANT_VISUALS_START -->

![Bioimage Nucleisegmentationboundarymodel - source-faithful ONNX run](assets/01-bioimage-nucleisegmentationboundarymodel-source-faithful-onnx-run.png)

![segmentation_logits channel_0](assets/02-segmentation-logits-channel-0.png)

![segmentation_logits channel_1](assets/03-segmentation-logits-channel-1.png)

![Runtime numeric evidence](assets/04-runtime-numeric-evidence.png)

![ONNX graph contract](assets/05-onnx-graph-contract.png)

![Runtime tensor evidence](assets/06-runtime-tensor-evidence.png)

![Input contract caveat](assets/07-input-contract-caveat.png)

<!-- BIOSIMULANT_VISUALS_END -->

## Caveat

Synthetic input validates ONNX execution only; segmentation accuracy needs source validation images.
