# PP-OCRv6 Detection Test

This directory contains a minimal Node.js test script for running the local
`PaddlePaddle/PP-OCRv6_medium_det_onnx` text detection model.

The script is intentionally not a CLI. Edit the paths at the top of
`test-ppocrv6-det.mjs`, then run it.

## Install

```bash
cd ppocrv6
npm install
```

## Configure

Open `test-ppocrv6-det.mjs` and fill:

```js
const INPUT_IMAGE_PATH = "";
const OUTPUT_JSON_PATH = "";
const DEBUG_IMAGE_PATH = "";
```

Example:

```js
const INPUT_IMAGE_PATH = "/tmp/ppocrv6-transformers-eval/sample.png";
const OUTPUT_JSON_PATH = "./boxes.json";
const DEBUG_IMAGE_PATH = "./debug.png";
```

`DEBUG_IMAGE_PATH` can stay empty if you only want JSON.

## Run

```bash
node test-ppocrv6-det.mjs
```

## Local Model

The script uses the checked-in local model files:

- `model/inference.onnx`
- `model/inference.yml`

It does not download models and does not call transformers.js pipeline APIs.

## Output

The JSON output contains:

- input image path
- model file paths
- ONNX Runtime input/output names and shapes
- preprocessing parameters
- postprocess thresholds
- detected text boxes

If `DEBUG_IMAGE_PATH` is set, the script also writes a copy of the input image
with detected boxes drawn on top.

## Notes

- This is text detection only, not text recognition.
- Runtime is `onnxruntime-web` with the WASM backend.
- Postprocess is a pure JS connected-components AABB approximation, not full
  PaddleOCR DBPostProcess.
