# Hf Emilyalsentzer Bio Clinicalbert

This Biosimulant lab wraps a source-derived ONNX artifact and reports runtime evidence from the bundled graph.

## Scientific Context

ClinicalBERT clinical-text embedding exported to ONNX.

## Source And Artifact

- `model`: https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT

- ONNX artifact: `models/core/artifacts/model.onnx`
- Runtime: ONNX Runtime CPU execution provider
- Validation scope: graph load, input/output contract, finite synthetic inference, Biosimulant wiring, and visual payloads

## Inputs

- `clinical_text_token_ids`: Inputs map to ONNX `input_ids` and `attention_mask`; tokenizer preprocessing is not bundled.
- `clinical_text_attention_mask`: Inputs map to ONNX `input_ids` and `attention_mask`; tokenizer preprocessing is not bundled.

The lab does not add raw image, raw text, raw protein sequence, tokenizer, or medical-image preprocessing unless that
preprocessing is bundled and validated. Public inputs are already-preprocessed tensors matching the ONNX graph contract.

## Outputs

- `sequence_embeddings`: Compact embedding evidence derived from the source ONNX graph output.
- `pooled_embedding`: Compact embedding evidence derived from the source ONNX graph output.

## Visualisations

The visualisation answers:

> What embedding tensors does the ClinicalBERT ONNX export emit for these already-tokenized clinical text inputs?

It shows provenance, graph schema, runtime tensor evidence, and a conservative caveat. Synthetic inference is a runtime
smoke test, not biological, clinical, or microscopy performance evidence.

<!-- BIOSIMULANT_VISUALS_START -->

![Hf Emilyalsentzer Bio Clinicalbert - source-faithful ONNX run](assets/01-hf-emilyalsentzer-bio-clinicalbert-source-faithful-onnx-run.png)

![pooled_embedding value heatmap](assets/02-pooled-embedding-value-heatmap.png)

![sequence_embeddings tensor_preview](assets/03-sequence-embeddings-tensor-preview.png)

![Runtime numeric evidence](assets/04-runtime-numeric-evidence.png)

![ONNX graph contract](assets/05-onnx-graph-contract.png)

![Runtime tensor evidence](assets/06-runtime-tensor-evidence.png)

![Input contract caveat](assets/07-input-contract-caveat.png)

<!-- BIOSIMULANT_VISUALS_END -->

## Caveat

No tokenizer is bundled, so public inputs are token tensors rather than raw text.
