# PP-OCRv6 Detection Test

This directory contains a minimal Node.js test script for running the local
`PaddlePaddle/PP-OCRv6_medium_det_onnx` text detection model.

The script is intentionally not a CLI. Edit the paths at the top of
`test-ppocrv6-det.cjs`, then run it.

## Install

```bash
cd ppocrv6
npm install
```

## Configure

Open `test-ppocrv6-det.cjs` and fill:

```js
const INPUT_IMAGE_PATH = "";
const OUTPUT_JSON_PATH = "";
const DEBUG_IMAGE_PATH = "";
```

Example:

```js
const INPUT_IMAGE_PATH = path.join(ASSETS_DIR, "sample.png");
const OUTPUT_JSON_PATH = path.join(ASSETS_DIR, "boxes.json");
const DEBUG_IMAGE_PATH = path.join(ASSETS_DIR, "debug.png");
```

`DEBUG_IMAGE_PATH` can stay empty if you only want JSON.

## Run

```bash
node test-ppocrv6-det.cjs
```

## Local Model

The script uses the local detection model files:

- `model/medium-det/inference.onnx`
- `model/medium-det/inference.yml`

Download model files with Make:

```bash
make medium-det
make medium-rec
make tiny-det
make tiny-rec
```

Or download all configured models:

```bash
make models
```

If direct Hugging Face access is unstable, pass a per-command proxy:

```bash
make models CURL_PROXY="-x http://127.0.0.1:7990"
```

The script does not call transformers.js pipeline APIs.

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
- The script uses CommonJS because `@techstark/opencv-js` is more stable through
  `require()` in this Node.js backend spike.
- Postprocess uses OpenCV.js contour/min-area-rect scoring plus `clipper-lib`
  polygon unclip to closely follow PaddleOCR DBPostProcess.
- The one-off PaddleOCR/Python golden comparison has already confirmed matching
  box counts and near-matching locations; the remaining differences are only
  pixel-level coordinate/score drift and occasional contour order changes.
