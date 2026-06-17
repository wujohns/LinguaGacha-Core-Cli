#!/usr/bin/env node
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import * as ort from "onnxruntime-web";
import sharp from "sharp";

// Fill these paths before running.
// Example:
//   const INPUT_IMAGE_PATH = "/tmp/ppocrv6-transformers-eval/sample.png";
//   const OUTPUT_JSON_PATH = "./boxes.json";
//   const DEBUG_IMAGE_PATH = "./debug.png";
const INPUT_IMAGE_PATH = "./test.png";
const OUTPUT_JSON_PATH = "./boxes.json";
const DEBUG_IMAGE_PATH = "./debug.png";

// Local PP-OCRv6 detection model files.
const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const MODEL_DIR = path.join(SCRIPT_DIR, "model");
const MODEL_ONNX_PATH = path.join(MODEL_DIR, "inference.onnx");
const MODEL_YML_PATH = path.join(MODEL_DIR, "inference.yml");

const DET_LIMIT_SIDE_LEN = 960;
const NORMALIZE_MEAN = [0.485, 0.456, 0.406];
const NORMALIZE_STD = [0.229, 0.224, 0.225];

const THRESH = 0.2;
const BOX_THRESH = 0.45;
const MAX_CANDIDATES = 3000;
const UNCLIP_RATIO = 1.4;
const MIN_SIZE = 3;

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
});

async function main() {
  if (!INPUT_IMAGE_PATH || !OUTPUT_JSON_PATH) {
    throw new Error("Please fill INPUT_IMAGE_PATH and OUTPUT_JSON_PATH at the top of this file.");
  }

  const modelBytes = await fs.readFile(MODEL_ONNX_PATH);
  const session = await ort.InferenceSession.create(modelBytes, {
    executionProviders: ["wasm"],
  });

  const inputName = session.inputNames[0] ?? "x";
  const outputName = session.outputNames[0] ?? "fetch_name_0";
  const image = await preprocessImage(INPUT_IMAGE_PATH);
  const inputTensor = new ort.Tensor("float32", image.tensorData, [
    1,
    3,
    image.height,
    image.width,
  ]);

  const outputs = await session.run({ [inputName]: inputTensor });
  const outputTensor = outputs[outputName] ?? outputs[session.outputNames[0]];
  if (!outputTensor) {
    throw new Error(`Model output not found. Available outputs: ${Object.keys(outputs).join(", ")}`);
  }

  const boxes = detectBoxes(outputTensor, image.originalWidth, image.originalHeight);
  const result = {
    image: path.resolve(INPUT_IMAGE_PATH),
    model: {
      onnx: MODEL_ONNX_PATH,
      yml: MODEL_YML_PATH,
    },
    runtime: {
      engine: "onnxruntime-web",
      backend: "wasm",
      inputName,
      outputName,
      inputShape: [1, 3, image.height, image.width],
      outputShape: outputTensor.dims,
    },
    preprocessing: {
      originalWidth: image.originalWidth,
      originalHeight: image.originalHeight,
      resizedWidth: image.width,
      resizedHeight: image.height,
      mean: NORMALIZE_MEAN,
      std: NORMALIZE_STD,
    },
    postprocess: {
      thresh: THRESH,
      boxThresh: BOX_THRESH,
      maxCandidates: MAX_CANDIDATES,
      unclipRatio: UNCLIP_RATIO,
      minSize: MIN_SIZE,
      mode: "connected-components-aabb",
    },
    boxes,
  };

  await fs.mkdir(path.dirname(path.resolve(OUTPUT_JSON_PATH)), { recursive: true });
  await fs.writeFile(OUTPUT_JSON_PATH, `${JSON.stringify(result, null, 2)}\n`, "utf8");

  if (DEBUG_IMAGE_PATH) {
    await drawDebugImage(INPUT_IMAGE_PATH, DEBUG_IMAGE_PATH, boxes);
  }

  console.log(`Wrote ${boxes.length} boxes to ${OUTPUT_JSON_PATH}`);
  if (DEBUG_IMAGE_PATH) {
    console.log(`Wrote debug image to ${DEBUG_IMAGE_PATH}`);
  }
}

async function preprocessImage(imagePath) {
  const metadata = await sharp(imagePath).metadata();
  if (!metadata.width || !metadata.height) {
    throw new Error(`Unable to read image dimensions: ${imagePath}`);
  }

  let width = metadata.width;
  let height = metadata.height;
  if (Math.max(width, height) > DET_LIMIT_SIDE_LEN) {
    const scale = DET_LIMIT_SIDE_LEN / Math.max(width, height);
    width = Math.round(width * scale);
    height = Math.round(height * scale);
  }
  width = Math.max(32, Math.round(width / 32) * 32);
  height = Math.max(32, Math.round(height / 32) * 32);

  const { data } = await sharp(imagePath)
    .resize(width, height, { fit: "fill" })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const tensorData = new Float32Array(3 * width * height);
  const planeSize = width * height;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const sourceIndex = (y * width + x) * 3;
      const targetIndex = y * width + x;
      const rgb = [
        data[sourceIndex] / 255,
        data[sourceIndex + 1] / 255,
        data[sourceIndex + 2] / 255,
      ];
      for (let channel = 0; channel < 3; channel += 1) {
        tensorData[channel * planeSize + targetIndex] =
          (rgb[channel] - NORMALIZE_MEAN[channel]) / NORMALIZE_STD[channel];
      }
    }
  }

  return {
    tensorData,
    width,
    height,
    originalWidth: metadata.width,
    originalHeight: metadata.height,
  };
}

function detectBoxes(outputTensor, originalWidth, originalHeight) {
  const [, , mapHeight, mapWidth] = outputTensor.dims;
  const scores = outputTensor.data;
  const bitmap = new Uint8Array(mapWidth * mapHeight);
  for (let index = 0; index < bitmap.length; index += 1) {
    bitmap[index] = scores[index] > THRESH ? 1 : 0;
  }

  const visited = new Uint8Array(bitmap.length);
  const queue = new Int32Array(bitmap.length);
  const components = [];
  const directions = [
    [-1, 0],
    [1, 0],
    [0, -1],
    [0, 1],
    [-1, -1],
    [1, -1],
    [-1, 1],
    [1, 1],
  ];

  for (let start = 0; start < bitmap.length; start += 1) {
    if (bitmap[start] === 0 || visited[start] !== 0) {
      continue;
    }

    let head = 0;
    let tail = 0;
    let xmin = mapWidth;
    let xmax = 0;
    let ymin = mapHeight;
    let ymax = 0;
    let pixelCount = 0;
    let scoreSum = 0;

    queue[tail] = start;
    tail += 1;
    visited[start] = 1;

    while (head < tail) {
      const current = queue[head];
      head += 1;
      const x = current % mapWidth;
      const y = Math.floor(current / mapWidth);

      xmin = Math.min(xmin, x);
      xmax = Math.max(xmax, x);
      ymin = Math.min(ymin, y);
      ymax = Math.max(ymax, y);
      pixelCount += 1;
      scoreSum += scores[current];

      for (const [dx, dy] of directions) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= mapWidth || ny >= mapHeight) {
          continue;
        }
        const next = ny * mapWidth + nx;
        if (bitmap[next] === 0 || visited[next] !== 0) {
          continue;
        }
        visited[next] = 1;
        queue[tail] = next;
        tail += 1;
      }
    }

    components.push({ xmin, xmax, ymin, ymax, pixelCount, score: scoreSum / pixelCount });
  }

  const boxes = [];
  components.sort((a, b) => b.pixelCount - a.pixelCount);

  for (const component of components.slice(0, MAX_CANDIDATES)) {
    const width = component.xmax - component.xmin + 1;
    const height = component.ymax - component.ymin + 1;
    if (Math.min(width, height) < MIN_SIZE || component.score < BOX_THRESH) {
      continue;
    }

    const distance = (width * height * UNCLIP_RATIO) / (2 * (width + height));
    const xmin = component.xmin - distance;
    const xmax = component.xmax + distance;
    const ymin = component.ymin - distance;
    const ymax = component.ymax + distance;

    boxes.push({
      points: [
        [clamp(Math.round((xmin / mapWidth) * originalWidth), 0, originalWidth), clamp(Math.round((ymin / mapHeight) * originalHeight), 0, originalHeight)],
        [clamp(Math.round((xmax / mapWidth) * originalWidth), 0, originalWidth), clamp(Math.round((ymin / mapHeight) * originalHeight), 0, originalHeight)],
        [clamp(Math.round((xmax / mapWidth) * originalWidth), 0, originalWidth), clamp(Math.round((ymax / mapHeight) * originalHeight), 0, originalHeight)],
        [clamp(Math.round((xmin / mapWidth) * originalWidth), 0, originalWidth), clamp(Math.round((ymax / mapHeight) * originalHeight), 0, originalHeight)],
      ],
      score: Math.round(component.score * 1_000_000) / 1_000_000,
    });
  }

  boxes.sort((a, b) => {
    const ay = (a.points[0][1] + a.points[2][1]) / 2;
    const by = (b.points[0][1] + b.points[2][1]) / 2;
    return Math.abs(ay - by) > 10 ? ay - by : a.points[0][0] - b.points[0][0];
  });
  return boxes;
}

async function drawDebugImage(imagePath, outputPath, boxes) {
  const metadata = await sharp(imagePath).metadata();
  if (!metadata.width || !metadata.height) {
    throw new Error(`Unable to read image dimensions: ${imagePath}`);
  }

  const overlays = boxes.map((box, index) => {
    const color = index % 2 === 0 ? "#00D084" : "#FF4D4F";
    const points = box.points.map(([x, y]) => `${x},${y}`).join(" ");
    const [labelX, labelY] = box.points[0];
    return `<polygon points="${points}" fill="none" stroke="${color}" stroke-width="3"/>
<text x="${labelX}" y="${Math.max(12, labelY - 4)}" fill="${color}" font-size="18" font-family="sans-serif">${index + 1}</text>`;
  });

  const svg = `<svg width="${metadata.width}" height="${metadata.height}" viewBox="0 0 ${metadata.width} ${metadata.height}" xmlns="http://www.w3.org/2000/svg">${overlays.join("")}</svg>`;
  await fs.mkdir(path.dirname(path.resolve(outputPath)), { recursive: true });
  await sharp(imagePath).composite([{ input: Buffer.from(svg), top: 0, left: 0 }]).toFile(outputPath);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}
