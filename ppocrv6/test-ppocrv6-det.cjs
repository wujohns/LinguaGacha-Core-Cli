#!/usr/bin/env node
const fs = require("node:fs/promises");
const path = require("node:path");

const ClipperLib = require("clipper-lib");
const cv = require("@techstark/opencv-js");
const ort = require("onnxruntime-web");
const sharp = require("sharp");
const yaml = require("js-yaml");

// Fill these paths before running.
// Example:
//   const INPUT_IMAGE_PATH = path.join(ASSETS_DIR, "sample.png");
//   const OUTPUT_JSON_PATH = path.join(ASSETS_DIR, "boxes.json");
//   const DEBUG_IMAGE_PATH = path.join(ASSETS_DIR, "debug.png");
const SCRIPT_DIR = __dirname;
const ASSETS_DIR = path.join(SCRIPT_DIR, "assets");
const INPUT_IMAGE_PATH = path.join(ASSETS_DIR, "comic_jp.png");
const OUTPUT_JSON_PATH = path.join(ASSETS_DIR, "boxes.json");
const DEBUG_IMAGE_PATH = path.join(ASSETS_DIR, "det_comic_jp.png");

// Local PP-OCRv6 detection model files.
const MODEL_DIR = path.join(SCRIPT_DIR, "model", "medium-det");
// const MODEL_DIR = path.join(SCRIPT_DIR, "model", "tiny-det");
const MODEL_ONNX_PATH = path.join(MODEL_DIR, "inference.onnx");
const MODEL_YML_PATH = path.join(MODEL_DIR, "inference.yml");

const MIN_SIZE = 3;
const CLIPPER_SCALE = 1024;

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
});

async function main() {
  if (!INPUT_IMAGE_PATH || !OUTPUT_JSON_PATH) {
    throw new Error("Please fill INPUT_IMAGE_PATH and OUTPUT_JSON_PATH at the top of this file.");
  }

  console.time("opencv");
  await waitForOpenCv();
  console.timeEnd("opencv");

  console.time("config");
  const config = await readInferenceConfig(MODEL_YML_PATH);
  console.timeEnd("config");

  console.time("session");
  const modelBytes = await fs.readFile(MODEL_ONNX_PATH);
  const session = await ort.InferenceSession.create(modelBytes, {
    executionProviders: ["wasm"],
  });
  console.timeEnd("session");

  const inputName = session.inputNames[0] ?? "x";
  const outputName = session.outputNames[0] ?? "fetch_name_0";

  console.time("preprocess");
  const image = await preprocessImage(INPUT_IMAGE_PATH, config.preprocessing);
  console.timeEnd("preprocess");

  const inputTensor = new ort.Tensor("float32", image.tensorData, [
    1,
    3,
    image.height,
    image.width,
  ]);

  console.time("inference");
  const outputs = await session.run({ [inputName]: inputTensor });
  console.timeEnd("inference");

  const outputTensor = outputs[outputName] ?? outputs[session.outputNames[0]];
  if (!outputTensor) {
    throw new Error(`Model output not found. Available outputs: ${Object.keys(outputs).join(", ")}`);
  }

  console.time("postprocess");
  const boxes = detectBoxes(outputTensor, image.originalWidth, image.originalHeight, config.postprocess);
  console.timeEnd("postprocess");

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
      ratioH: image.ratioH,
      ratioW: image.ratioW,
      imgMode: config.preprocessing.imgMode,
      scale: config.preprocessing.scale,
      mean: config.preprocessing.mean,
      std: config.preprocessing.std,
      detResizeForTest: config.preprocessing.detResizeForTest,
    },
    postprocess: {
      ...config.postprocess,
      mode: "opencv-dbpostprocess-quad",
    },
    boxes,
  };

  console.time("write-json");
  await fs.mkdir(path.dirname(path.resolve(OUTPUT_JSON_PATH)), { recursive: true });
  await fs.writeFile(OUTPUT_JSON_PATH, `${JSON.stringify(result, null, 2)}\n`, "utf8");
  console.timeEnd("write-json");

  if (DEBUG_IMAGE_PATH) {
    console.time("debug-image");
    await drawDebugImage(INPUT_IMAGE_PATH, DEBUG_IMAGE_PATH, boxes);
    console.timeEnd("debug-image");
  }

  console.log(`Wrote ${boxes.length} boxes to ${OUTPUT_JSON_PATH}`);
  if (DEBUG_IMAGE_PATH) {
    console.log(`Wrote debug image to ${DEBUG_IMAGE_PATH}`);
  }
}

async function waitForOpenCv() {
  const deadline = Date.now() + 10_000;
  while (!cv.Mat && Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, 25));
  }
  if (!cv.Mat) {
    throw new Error("OpenCV.js runtime was not initialized in time.");
  }
}

async function readInferenceConfig(ymlPath) {
  const config = yaml.load(await fs.readFile(ymlPath, "utf8"));
  const transforms = config?.PreProcess?.transform_ops ?? [];
  const decodeImage = findTransform(transforms, "DecodeImage") ?? {};
  const resize = findTransform(transforms, "DetResizeForTest");
  const normalize = findTransform(transforms, "NormalizeImage") ?? {};
  const postprocess = config?.PostProcess ?? {};

  return {
    preprocessing: {
      imgMode: String(decodeImage.img_mode ?? "RGB").toUpperCase(),
      scale: parseScale(normalize.scale ?? "1./255."),
      mean: normalize.mean ?? [0.485, 0.456, 0.406],
      std: normalize.std ?? [0.229, 0.224, 0.225],
      detResizeForTest: normalizeDetResizeConfig(resize),
    },
    postprocess: {
      thresh: Number(postprocess.thresh ?? 0.3),
      boxThresh: Number(postprocess.box_thresh ?? 0.7),
      maxCandidates: Number(postprocess.max_candidates ?? 1000),
      unclipRatio: Number(postprocess.unclip_ratio ?? 2.0),
      minSize: MIN_SIZE,
    },
  };
}

function findTransform(transforms, name) {
  const entry = transforms.find((transform) => Object.hasOwn(transform, name));
  return entry ? entry[name] : undefined;
}

function parseScale(value) {
  if (typeof value === "number") {
    return value;
  }
  if (typeof value !== "string") {
    return 1 / 255;
  }
  const normalized = value.replace(/\s+/g, "").replace(/\.$/, "");
  const fraction = normalized.match(/^([0-9.]+)\/([0-9.]+)$/);
  if (fraction) {
    return Number(fraction[1]) / Number(fraction[2]);
  }
  return Number(normalized);
}

function normalizeDetResizeConfig(config) {
  if (!config) {
    return {
      resizeType: 0,
      limitSideLen: 736,
      limitType: "min",
      maxSideLimit: 4000,
    };
  }
  if (config.image_shape) {
    return {
      resizeType: 1,
      imageShape: config.image_shape,
      keepRatio: Boolean(config.keep_ratio),
      maxSideLimit: Number(config.max_side_limit ?? 4000),
    };
  }
  if (config.resize_long) {
    return {
      resizeType: 2,
      resizeLong: Number(config.resize_long),
      maxSideLimit: Number(config.max_side_limit ?? 4000),
    };
  }
  return {
    resizeType: 0,
    limitSideLen: Number(config.limit_side_len ?? 736),
    limitType: config.limit_type ?? "min",
    maxSideLimit: Number(config.max_side_limit ?? 4000),
  };
}

async function preprocessImage(imagePath, preprocessing) {
  const metadata = await sharp(imagePath).metadata();
  if (!metadata.width || !metadata.height) {
    throw new Error(`Unable to read image dimensions: ${imagePath}`);
  }

  const size = resizeForTest(metadata.width, metadata.height, preprocessing.detResizeForTest);
  const { data } = await sharp(imagePath)
    .resize(size.width, size.height, { fit: "fill" })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const tensorData = new Float32Array(3 * size.width * size.height);
  const planeSize = size.width * size.height;
  for (let y = 0; y < size.height; y += 1) {
    for (let x = 0; x < size.width; x += 1) {
      const sourceIndex = (y * size.width + x) * 3;
      const targetIndex = y * size.width + x;
      const r = data[sourceIndex];
      const g = data[sourceIndex + 1];
      const b = data[sourceIndex + 2];
      const channels = preprocessing.imgMode === "BGR" ? [b, g, r] : [r, g, b];
      for (let channel = 0; channel < 3; channel += 1) {
        tensorData[channel * planeSize + targetIndex] =
          (channels[channel] * preprocessing.scale - preprocessing.mean[channel]) / preprocessing.std[channel];
      }
    }
  }

  return {
    tensorData,
    width: size.width,
    height: size.height,
    ratioW: size.ratioW,
    ratioH: size.ratioH,
    originalWidth: metadata.width,
    originalHeight: metadata.height,
  };
}

function resizeForTest(originalWidth, originalHeight, config) {
  if (originalWidth + originalHeight < 64) {
    originalWidth = Math.max(32, originalWidth);
    originalHeight = Math.max(32, originalHeight);
  }

  if (config.resizeType === 1) {
    let [height, width] = config.imageShape;
    if (config.keepRatio) {
      width = Math.ceil((originalWidth * height) / originalHeight / 32) * 32;
    }
    return {
      width,
      height,
      ratioW: width / originalWidth,
      ratioH: height / originalHeight,
    };
  }

  if (config.resizeType === 2) {
    const ratio = config.resizeLong / Math.max(originalHeight, originalWidth);
    const maxStride = 128;
    const height = Math.ceil(Math.trunc(originalHeight * ratio) / maxStride) * maxStride;
    const width = Math.ceil(Math.trunc(originalWidth * ratio) / maxStride) * maxStride;
    return {
      width,
      height,
      ratioW: width / originalWidth,
      ratioH: height / originalHeight,
    };
  }

  let ratio = 1;
  if (config.limitType === "max" && Math.max(originalHeight, originalWidth) > config.limitSideLen) {
    ratio = config.limitSideLen / Math.max(originalHeight, originalWidth);
  } else if (config.limitType === "min" && Math.min(originalHeight, originalWidth) < config.limitSideLen) {
    ratio = config.limitSideLen / Math.min(originalHeight, originalWidth);
  } else if (config.limitType === "resize_long") {
    ratio = config.limitSideLen / Math.max(originalHeight, originalWidth);
  }

  let height = Math.trunc(originalHeight * ratio);
  let width = Math.trunc(originalWidth * ratio);
  if (Math.max(height, width) > config.maxSideLimit) {
    const maxRatio = config.maxSideLimit / Math.max(height, width);
    height = Math.trunc(height * maxRatio);
    width = Math.trunc(width * maxRatio);
  }

  height = Math.max(Math.round(height / 32) * 32, 32);
  width = Math.max(Math.round(width / 32) * 32, 32);
  return {
    width,
    height,
    ratioW: width / originalWidth,
    ratioH: height / originalHeight,
  };
}

function detectBoxes(outputTensor, originalWidth, originalHeight, postprocess) {
  const [, , mapHeight, mapWidth] = outputTensor.dims;
  const scores = outputTensor.data;
  const mask = new cv.Mat(mapHeight, mapWidth, cv.CV_8UC1);
  for (let index = 0; index < mask.data.length; index += 1) {
    mask.data[index] = scores[index] > postprocess.thresh ? 255 : 0;
  }

  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  const boxes = [];

  try {
    cv.findContours(mask, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);
    const count = Math.min(contours.size(), postprocess.maxCandidates);
    for (let index = 0; index < count; index += 1) {
      const contour = contours.get(index);
      try {
        const { points, shortSide } = getMiniBox(contour);
        if (shortSide < postprocess.minSize) {
          continue;
        }

        const score = boxScoreFast(scores, mapWidth, mapHeight, points);
        if (score < postprocess.boxThresh) {
          continue;
        }

        const expanded = unclip(points, postprocess.unclipRatio);
        if (expanded.length === 0) {
          continue;
        }

        const expandedContour = pointsToMat(expanded);
        try {
          const expandedBox = getMiniBox(expandedContour);
          if (expandedBox.shortSide < postprocess.minSize + 2) {
            continue;
          }

          boxes.push({
            points: expandedBox.points.map(({ x, y }) => [
              clamp(Math.round((x / mapWidth) * originalWidth), 0, originalWidth),
              clamp(Math.round((y / mapHeight) * originalHeight), 0, originalHeight),
            ]),
            score: Math.round(score * 1_000_000) / 1_000_000,
          });
        } finally {
          expandedContour.delete();
        }
      } finally {
        contour.delete();
      }
    }
  } finally {
    mask.delete();
    contours.delete();
    hierarchy.delete();
  }

  return boxes;
}

function getMiniBox(contour) {
  const rect = cv.minAreaRect(contour);
  const vertices = cv.RotatedRect.points(rect);
  const points = [...vertices].sort((a, b) => (a.x === b.x ? a.y - b.y : a.x - b.x));
  const [leftA, leftB, rightA, rightB] = points;
  return {
    points: [
      leftA.y <= leftB.y ? leftA : leftB,
      rightA.y <= rightB.y ? rightA : rightB,
      rightA.y <= rightB.y ? rightB : rightA,
      leftA.y <= leftB.y ? leftB : leftA,
    ],
    shortSide: Math.min(rect.size.width, rect.size.height),
  };
}

function boxScoreFast(scores, width, height, points) {
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const xmin = clamp(Math.floor(Math.min(...xs)), 0, width - 1);
  const xmax = clamp(Math.ceil(Math.max(...xs)), 0, width - 1);
  const ymin = clamp(Math.floor(Math.min(...ys)), 0, height - 1);
  const ymax = clamp(Math.ceil(Math.max(...ys)), 0, height - 1);
  const mask = cv.Mat.zeros(ymax - ymin + 1, xmax - xmin + 1, cv.CV_8UC1);
  const roi = new cv.Mat(ymax - ymin + 1, xmax - xmin + 1, cv.CV_32FC1);
  const polygon = pointsToMat(
    points.map((point) => ({ x: point.x - xmin, y: point.y - ymin })),
    true,
  );
  const polygons = new cv.MatVector();

  try {
    polygons.push_back(polygon);
    cv.fillPoly(mask, polygons, new cv.Scalar(1));
    for (let y = ymin; y <= ymax; y += 1) {
      for (let x = xmin; x <= xmax; x += 1) {
        roi.floatPtr(y - ymin, x - xmin)[0] = scores[y * width + x];
      }
    }
    return cv.mean(roi, mask)[0];
  } finally {
    mask.delete();
    roi.delete();
    polygon.delete();
    polygons.delete();
  }
}

function unclip(points, unclipRatio) {
  const area = polygonArea(points);
  const perimeter = polygonPerimeter(points);
  if (area <= 0 || perimeter <= 0) {
    return [];
  }

  const offset = new ClipperLib.ClipperOffset();
  offset.AddPath(
    points.map((point) => ({
      X: Math.round(point.x * CLIPPER_SCALE),
      Y: Math.round(point.y * CLIPPER_SCALE),
    })),
    ClipperLib.JoinType.jtRound,
    ClipperLib.EndType.etClosedPolygon,
  );

  const expanded = [];
  offset.Execute(expanded, (area * unclipRatio * CLIPPER_SCALE) / perimeter);
  if (expanded.length !== 1) {
    return [];
  }
  return expanded[0].map((point) => ({ x: point.X / CLIPPER_SCALE, y: point.Y / CLIPPER_SCALE }));
}

function pointsToMat(points, int32 = false) {
  const mat = new cv.Mat(points.length, 1, int32 ? cv.CV_32SC2 : cv.CV_32FC2);
  const data = int32 ? mat.data32S : mat.data32F;
  for (let index = 0; index < points.length; index += 1) {
    data[index * 2] = int32 ? Math.round(points[index].x) : points[index].x;
    data[index * 2 + 1] = int32 ? Math.round(points[index].y) : points[index].y;
  }
  return mat;
}

function polygonArea(points) {
  let area = 0;
  for (let index = 0; index < points.length; index += 1) {
    const current = points[index];
    const next = points[(index + 1) % points.length];
    area += current.x * next.y - current.y * next.x;
  }
  return Math.abs(area) / 2;
}

function polygonPerimeter(points) {
  let perimeter = 0;
  for (let index = 0; index < points.length; index += 1) {
    const current = points[index];
    const next = points[(index + 1) % points.length];
    perimeter += Math.hypot(next.x - current.x, next.y - current.y);
  }
  return perimeter;
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
