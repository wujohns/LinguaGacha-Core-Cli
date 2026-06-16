# PP-OCRv6 Medium Detection ONNX + Transformers.js Evaluation

This is a standalone Node.js spike for evaluating whether
`PaddlePaddle/PP-OCRv6_medium_det_onnx` can be used from transformers.js, and
for running the ONNX text detection model with PaddleOCR-style pre/post
processing.

It is intentionally independent from the rest of the LinguaGacha codebase.

## Install

```bash
cd ppocrv6
npm install
```

If Hugging Face downloads are slow or unstable in this environment, use the
local proxy:

```bash
HTTPS_PROXY=http://127.0.0.1:7990 HTTP_PROXY=http://127.0.0.1:7990 npm install
```

## Run

```bash
node test-ppocrv6-det.mjs \
  --image ./sample.png \
  --out ./boxes.json \
  --debug-image ./debug.png
```

The default model path is the checked-in local model directory:

`./model`

That directory contains:

- `inference.onnx`
- `inference.yml`

With the local proxy, only needed if you pass a remote Hugging Face model id:

```bash
HTTPS_PROXY=http://127.0.0.1:7990 HTTP_PROXY=http://127.0.0.1:7990 \
  node test-ppocrv6-det.mjs --image ./sample.png --out ./boxes.json --debug-image ./debug.png
```

The original Hugging Face model is:

`PaddlePaddle/PP-OCRv6_medium_det_onnx`

The script dynamically tries to import `@huggingface/transformers` for the
pipeline compatibility probe. It is intentionally not installed by default in
this spike because recent transformers.js packages pull `onnxruntime-node`,
whose native installer can fail in this environment when its download endpoint
returns a redirect. If you want to test the real import anyway, run:

```bash
npm install @huggingface/transformers
```

You can pass either a local model directory containing `inference.onnx` and
`inference.yml`, a local `.onnx` file, or a Hugging Face model id:

```bash
node test-ppocrv6-det.mjs --image ./sample.png --model ./model --out ./boxes.json
node test-ppocrv6-det.mjs --image ./sample.png --model ./model-dir --out ./boxes.json
node test-ppocrv6-det.mjs --image ./sample.png --model ./inference.onnx --out ./boxes.json
```

## What It Proves

- `transformers.js` pipeline compatibility is checked first and recorded in the
  JSON output. This model is expected not to work as a direct pipeline model.
- Actual inference is run through `onnxruntime-web` WASM because this PaddleOCR
  ONNX file does not come with the transformers.js pipeline/model config
  structure. This still validates the ONNX graph from JavaScript.
- The postprocess uses a pure JS connected-components AABB approximation. It is
  enough for this spike, but it is not a full replacement for PaddleOCR's
  OpenCV-based rotated DBPostProcess.
- The script outputs text detection boxes only. It does not recognize text.
