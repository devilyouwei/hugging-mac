# PP-DocLayoutV3

This package integrates the Apache-2.0 licensed
[`PaddlePaddle/PP-DocLayoutV3_safetensors`](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_safetensors)
checkpoint as the canonical SDK model `paddlepaddle/pp-doclayout-v3`.

It exposes `DocumentLayoutAnalysis`, returning semantic regions, source-image
bounding boxes, optional polygons, confidence scores, and model-predicted
reading order. The fixed 800×800 RGB preprocessing and label map follow the
upstream checkpoint. PyTorch/MPS and Core ML share the same SDK postprocessor.

## Resource flow

The Models page and `ModelResourceService` use `resources.py` to:

1. download the three pinned Hugging Face source files into the canonical
   `base/pytorch-mps/source` artifact;
2. call `PPDocLayoutV3Converter` from
   `hugging_mac_sdk/models/pp_doclayout_v3/converter.py`;
3. atomically publish `model.mlpackage` plus `metadata.json` under
   `base/coreml/coreml-fp16`.

The converter accepts only this model's complete safetensors source artifact
and produces an ML Program with the fixed `pixel_values` input shape
`[1, 3, 800, 800]`. It exports raw `logits`, `pred_boxes`, `order_logits`, and
`out_masks`; thresholding, reading-order ranking, coordinate restoration, and
polygon extraction deliberately remain in `utils/postprocess.py`.

Conversion is initiated through the SDK resource conversion API with target
format `coreml`; `converter.py` is the single model-specific conversion entry,
not an ad-hoc repository script.

Install dependencies with the `layout` extra before source inference or
conversion.
