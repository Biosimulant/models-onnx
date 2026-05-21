Imported transformer artifact for `SciBERT`.

Status:
- `model.onnx` is a validated local ONNX export produced with `transformers.onnx`.
- The exported graph expects `input_ids`, `attention_mask`, and `token_type_ids`.
- The second raw ONNX output name was normalized to `pooled_output`.
- Export details are recorded in `export-metadata.json`.
