#!/usr/bin/env node
const fs = require("node:fs/promises");
const path = require("node:path");

const ort = require("onnxruntime-web");
const sharp = require("sharp");

const SCRIPT_DIR = __dirname;
const ASSETS_DIR = path.join(SCRIPT_DIR, "assets");
const MODEL_DIR = path.join(SCRIPT_DIR, "model", "comic-bubble");

const MODEL_ONNX_PATH = path.join(MODEL_DIR, "detector-v4-s_int8.onnx");
const MODEL_CONFIG_PATH = path.join(MODEL_DIR, "config.json");
const PREPROCESSOR_CONFIG_PATH = path.join(MODEL_DIR, "preprocessor_config.json");

const INPUT_IMAGE_PATHS = [
  path.join(ASSETS_DIR, "comic_en.png"),
  path.join(ASSETS_DIR, "comic_jp.png"),
];

const SCORE_THRESHOLD = Number(process.env.COMIC_BUBBLE_THRESHOLD ?? "0.35");
const LABEL_FILTER = (process.env.COMIC_BUBBLE_LABELS ?? "")
  .split(",")
  .map((label) => label.trim())
  .filter(Boolean);

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
});

async function main() {
  const [modelConfig, preprocessorConfig] = await Promise.all([
    readJson(MODEL_CONFIG_PATH),
    readJson(PREPROCESSOR_CONFIG_PATH),
    assertFile(MODEL_ONNX_PATH),
  ]);

  const imageSize = normalizeImageSize(preprocessorConfig);
  const id2label = normalizeId2Label(modelConfig.id2label);
  const labelFilter = new Set(LABEL_FILTER);

  console.time("session");
  const modelBytes = await fs.readFile(MODEL_ONNX_PATH);
  const session = await ort.InferenceSession.create(modelBytes, {
    executionProviders: ["wasm"],
  });
  console.timeEnd("session");

  const inputName = session.inputNames.includes("images") ? "images" : session.inputNames[0];
  const targetSizeInputName = session.inputNames.includes("orig_target_sizes")
    ? "orig_target_sizes"
    : session.inputNames[1];

  if (!inputName || !targetSizeInputName) {
    throw new Error(`Unexpected model inputs: ${session.inputNames.join(", ")}`);
  }

  for (const imagePath of INPUT_IMAGE_PATHS) {
    await runImage({
      imagePath,
      session,
      inputName,
      targetSizeInputName,
      imageSize,
      id2label,
      labelFilter,
      preprocessorConfig,
    });
  }
}

async function runImage({
  imagePath,
  session,
  inputName,
  targetSizeInputName,
  imageSize,
  id2label,
  labelFilter,
  preprocessorConfig,
}) {
  const imageBaseName = path.basename(imagePath, path.extname(imagePath));
  const outputJsonPath = path.join(ASSETS_DIR, `${imageBaseName}.comic-bubble.json`);
  const debugImagePath = path.join(ASSETS_DIR, `${imageBaseName}.comic-bubble.png`);

  console.log(`\n${path.relative(SCRIPT_DIR, imagePath)}`);

  console.time(`${imageBaseName}:preprocess`);
  const image = await preprocessImage(imagePath, imageSize, preprocessorConfig);
  console.timeEnd(`${imageBaseName}:preprocess`);

  const imageTensor = new ort.Tensor("float32", image.tensorData, [
    1,
    3,
    imageSize.height,
    imageSize.width,
  ]);
  const targetSizeTensor = new ort.Tensor("int64", BigInt64Array.from([
    BigInt(image.originalWidth),
    BigInt(image.originalHeight),
  ]), [1, 2]);

  console.time(`${imageBaseName}:inference`);
  const outputs = await session.run({
    [inputName]: imageTensor,
    [targetSizeInputName]: targetSizeTensor,
  });
  console.timeEnd(`${imageBaseName}:inference`);

  const detections = decodeDetections(outputs, id2label, image.originalWidth, image.originalHeight)
    .filter((detection) => detection.score >= SCORE_THRESHOLD)
    .filter((detection) => labelFilter.size === 0 || labelFilter.has(detection.label))
    .map((detection, index) => ({
      index,
      ...detection,
      bbox: detection.bbox.map((value) => round(value, 3)),
    }));

  const result = {
    image: path.resolve(imagePath),
    model: {
      repository: "ogkalu/comic-text-and-bubble-detector",
      variant: "detector-v4-s_int8.onnx",
      onnx: MODEL_ONNX_PATH,
      config: MODEL_CONFIG_PATH,
      preprocessor: PREPROCESSOR_CONFIG_PATH,
    },
    runtime: {
      engine: "onnxruntime-web",
      backend: "wasm",
      inputNames: session.inputNames,
      outputNames: session.outputNames,
      inputShape: [1, 3, imageSize.height, imageSize.width],
    },
    preprocessing: {
      originalWidth: image.originalWidth,
      originalHeight: image.originalHeight,
      resizedWidth: imageSize.width,
      resizedHeight: imageSize.height,
      doRescale: Boolean(preprocessorConfig.do_rescale),
      rescaleFactor: Number(preprocessorConfig.rescale_factor ?? 1 / 255),
      doNormalize: Boolean(preprocessorConfig.do_normalize),
    },
    postprocess: {
      threshold: SCORE_THRESHOLD,
      labelFilter: [...labelFilter],
      boxFormat: "xyxy",
    },
    counts: countByLabel(detections),
    detections,
  };

  await fs.writeFile(outputJsonPath, `${JSON.stringify(result, null, 2)}\n`, "utf8");
  await drawDebugImage(imagePath, debugImagePath, detections);

  console.log(`detections: ${detections.length}`, result.counts);
  console.log(`json: ${path.relative(SCRIPT_DIR, outputJsonPath)}`);
  console.log(`debug: ${path.relative(SCRIPT_DIR, debugImagePath)}`);
}

async function preprocessImage(imagePath, imageSize, preprocessorConfig) {
  const metadata = await sharp(imagePath).metadata();
  if (!metadata.width || !metadata.height) {
    throw new Error(`Unable to read image dimensions: ${imagePath}`);
  }

  const { data } = await sharp(imagePath)
    .resize(imageSize.width, imageSize.height, { fit: "fill" })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const doRescale = Boolean(preprocessorConfig.do_rescale);
  const rescaleFactor = Number(preprocessorConfig.rescale_factor ?? 1 / 255);
  const doNormalize = Boolean(preprocessorConfig.do_normalize);
  const mean = preprocessorConfig.image_mean ?? [0.485, 0.456, 0.406];
  const std = preprocessorConfig.image_std ?? [0.229, 0.224, 0.225];
  const planeSize = imageSize.width * imageSize.height;
  const tensorData = new Float32Array(3 * planeSize);

  for (let y = 0; y < imageSize.height; y += 1) {
    for (let x = 0; x < imageSize.width; x += 1) {
      const sourceIndex = (y * imageSize.width + x) * 3;
      const targetIndex = y * imageSize.width + x;
      const values = [data[sourceIndex], data[sourceIndex + 1], data[sourceIndex + 2]];
      for (let channel = 0; channel < 3; channel += 1) {
        let value = values[channel];
        if (doRescale) {
          value *= rescaleFactor;
        }
        if (doNormalize) {
          value = (value - mean[channel]) / std[channel];
        }
        tensorData[channel * planeSize + targetIndex] = value;
      }
    }
  }

  return {
    tensorData,
    originalWidth: metadata.width,
    originalHeight: metadata.height,
  };
}

function decodeDetections(outputs, id2label, imageWidth, imageHeight) {
  const labelsTensor = outputs.labels;
  const boxesTensor = outputs.boxes;
  const scoresTensor = outputs.scores;

  if (!labelsTensor || !boxesTensor || !scoresTensor) {
    throw new Error(`Unexpected model outputs: ${Object.keys(outputs).join(", ")}`);
  }

  const detections = [];
  for (let index = 0; index < scoresTensor.data.length; index += 1) {
    const labelId = Number(labelsTensor.data[index]);
    const boxOffset = index * 4;
    const bbox = [
      Number(boxesTensor.data[boxOffset]),
      Number(boxesTensor.data[boxOffset + 1]),
      Number(boxesTensor.data[boxOffset + 2]),
      Number(boxesTensor.data[boxOffset + 3]),
    ];
    const clampedBbox = [
      clamp(bbox[0], 0, imageWidth),
      clamp(bbox[1], 0, imageHeight),
      clamp(bbox[2], 0, imageWidth),
      clamp(bbox[3], 0, imageHeight),
    ];
    detections.push({
      labelId,
      label: id2label[labelId] ?? String(labelId),
      score: round(Number(scoresTensor.data[index]), 6),
      bbox: clampedBbox,
    });
  }

  return detections.sort((a, b) => b.score - a.score);
}

async function drawDebugImage(imagePath, outputPath, detections) {
  const metadata = await sharp(imagePath).metadata();
  if (!metadata.width || !metadata.height) {
    throw new Error(`Unable to read image dimensions: ${imagePath}`);
  }

  const overlays = detections.map((detection, index) => {
    const [x1, y1, x2, y2] = detection.bbox;
    const width = Math.max(0, x2 - x1);
    const height = Math.max(0, y2 - y1);
    const color = labelColor(detection.label);
    const text = `${index + 1} ${escapeXml(detection.label)} ${detection.score.toFixed(2)}`;
    const labelY = Math.max(14, y1 - 4);
    return `<rect x="${x1}" y="${y1}" width="${width}" height="${height}" fill="none" stroke="${color}" stroke-width="3"/>
<rect x="${x1}" y="${labelY - 14}" width="${Math.max(58, text.length * 8)}" height="17" fill="${color}" fill-opacity="0.85"/>
<text x="${x1 + 3}" y="${labelY - 2}" fill="#ffffff" font-size="12" font-family="sans-serif">${text}</text>`;
  });

  const svg = `<svg width="${metadata.width}" height="${metadata.height}" viewBox="0 0 ${metadata.width} ${metadata.height}" xmlns="http://www.w3.org/2000/svg">${overlays.join("")}</svg>`;
  await sharp(imagePath).composite([{ input: Buffer.from(svg), top: 0, left: 0 }]).toFile(outputPath);
}

async function readJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, "utf8"));
}

async function assertFile(filePath) {
  try {
    await fs.access(filePath);
  } catch {
    throw new Error(`Missing file: ${filePath}. Run "make comic-bubble" in ppocrv6 first.`);
  }
}

function normalizeImageSize(config) {
  const width = Number(config?.size?.width ?? 640);
  const height = Number(config?.size?.height ?? 640);
  return { width, height };
}

function normalizeId2Label(id2label) {
  const labels = {};
  for (const [key, value] of Object.entries(id2label ?? {})) {
    labels[Number(key)] = String(value);
  }
  return labels;
}

function countByLabel(detections) {
  return detections.reduce((counts, detection) => {
    counts[detection.label] = (counts[detection.label] ?? 0) + 1;
    return counts;
  }, {});
}

function labelColor(label) {
  if (label === "bubble") {
    return "#00A36C";
  }
  if (label === "text_bubble") {
    return "#2563EB";
  }
  return "#D97706";
}

function escapeXml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function round(value, digits) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}
