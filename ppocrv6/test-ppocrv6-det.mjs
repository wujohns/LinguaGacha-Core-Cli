#!/usr/bin/env node
import fs from "node:fs/promises";
import fsSync from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import * as ort from "onnxruntime-web";
import sharp from "sharp";
import { ProxyAgent, fetch as undiciFetch } from "undici";

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_HF_MODEL = "PaddlePaddle/PP-OCRv6_medium_det_onnx";
const DEFAULT_MODEL = path.join(SCRIPT_DIR, "model");
const DEFAULT_CACHE_DIR = path.join(SCRIPT_DIR, ".model-cache");
const HF_BASE_URL = "https://huggingface.co";
const DET_LIMIT_SIDE_LEN = 960;
const DET_LIMIT_TYPE = "max";

const POSTPROCESS_DEFAULTS = {
  thresh: 0.2,
  boxThresh: 0.45,
  maxCandidates: 3000,
  unclipRatio: 1.4,
  minSize: 3,
  mode: "connected-components-aabb",
};

const NORMALIZE = {
  mean: [0.485, 0.456, 0.406],
  std: [0.229, 0.224, 0.225],
};

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
});

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    process.stdout.write(buildHelp());
    return;
  }
  if (!options.image) {
    throw new Error("Missing required option --image");
  }
  if (!options.out) {
    throw new Error("Missing required option --out");
  }

  const modelRef = options.model ?? DEFAULT_MODEL;
  const modelFiles = await resolveModelFiles(modelRef, options.cacheDir ?? DEFAULT_CACHE_DIR);

  const pipelineCheck = options.skipPipelineCheck
    ? { attempted: false, ok: false, error: "Skipped by --skip-pipeline-check" }
    : await checkTransformersPipeline(modelRef, modelFiles);

  const modelBytes = await fs.readFile(modelFiles.onnxPath);
  const session = await ort.InferenceSession.create(modelBytes, {
    executionProviders: ["wasm"],
  });
  const input = session.inputNames[0] ?? "x";
  const output = session.outputNames[0] ?? "fetch_name_0";

  const preprocessed = await preprocessImage(options.image);
  const tensor = new ort.Tensor("float32", preprocessed.data, [
    1,
    3,
    preprocessed.height,
    preprocessed.width,
  ]);
  const results = await session.run({ [input]: tensor });
  const outputTensor = results[output] ?? results[session.outputNames[0]];
  if (!outputTensor) {
    throw new Error(`Model output not found. Available outputs: ${Object.keys(results).join(", ")}`);
  }

  const boxes = postprocessDetection(outputTensor, preprocessed.shape);
  const payload = {
    image: path.resolve(options.image),
    model: modelRef,
    files: {
      onnx: modelFiles.onnxPath,
      yml: modelFiles.ymlPath,
    },
    transformersPipeline: pipelineCheck,
    runtime: {
      engine: "onnxruntime-web",
      backend: "wasm",
      inputName: input,
      outputName: output,
      inputShape: [1, 3, preprocessed.height, preprocessed.width],
      outputShape: outputTensor.dims,
    },
    preprocessing: {
      resize: {
        originalWidth: preprocessed.shape.srcW,
        originalHeight: preprocessed.shape.srcH,
        resizedWidth: preprocessed.width,
        resizedHeight: preprocessed.height,
        ratioW: preprocessed.shape.ratioW,
        ratioH: preprocessed.shape.ratioH,
      },
      normalize: NORMALIZE,
    },
    postprocess: POSTPROCESS_DEFAULTS,
    boxes,
  };

  await fs.mkdir(path.dirname(path.resolve(options.out)), { recursive: true });
  await fs.writeFile(options.out, `${JSON.stringify(payload, null, 2)}\n`, "utf8");

  if (options.debugImage) {
    await drawDebugImage(options.image, options.debugImage, boxes);
  }

  console.log(`Wrote ${boxes.length} boxes to ${options.out}`);
  if (options.debugImage) {
    console.log(`Wrote debug image to ${options.debugImage}`);
  }
  console.log(
    `transformers.js pipeline direct load: ${
      pipelineCheck.ok ? "OK" : `not usable (${pipelineCheck.error})`
    }`,
  );
}

function parseArgs(argv) {
  const options = {
    image: "",
    model: DEFAULT_MODEL,
    out: "",
    debugImage: "",
    cacheDir: DEFAULT_CACHE_DIR,
    skipPipelineCheck: false,
    help: false,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    const value = argv[index + 1];
    if (token === "--help" || token === "-h") {
      options.help = true;
    } else if (token === "--image") {
      options.image = readValue(token, value);
      index += 1;
    } else if (token === "--model") {
      options.model = readValue(token, value);
      index += 1;
    } else if (token === "--out") {
      options.out = readValue(token, value);
      index += 1;
    } else if (token === "--debug-image") {
      options.debugImage = readValue(token, value);
      index += 1;
    } else if (token === "--cache-dir") {
      options.cacheDir = readValue(token, value);
      index += 1;
    } else if (token === "--skip-pipeline-check") {
      options.skipPipelineCheck = true;
    } else {
      throw new Error(`Unknown option: ${token}`);
    }
  }
  return options;
}

function readValue(name, value) {
  if (!value || value.startsWith("--")) {
    throw new Error(`Missing value for ${name}`);
  }
  return value;
}

function buildHelp() {
  return `Usage:
  node test-ppocrv6-det.mjs --image <path> --out <path> [--debug-image <path>]

Options:
  --image                 Required input image path
  --out                   Required JSON output path
  --debug-image           Optional image with detected boxes overlaid
  --model                 HF model id, local model directory, or local .onnx file
                          Default: ${DEFAULT_MODEL}
  --cache-dir             Model cache directory
                          Default: ${DEFAULT_CACHE_DIR}
  --skip-pipeline-check   Skip the transformers.js pipeline compatibility probe
  --help                  Show this help
`;
}

async function resolveModelFiles(modelRef, cacheDir) {
  const absolute = path.resolve(modelRef);
  if (fsSync.existsSync(absolute)) {
    const stat = await fs.stat(absolute);
    if (stat.isDirectory()) {
      const onnxPath = path.join(absolute, "inference.onnx");
      const ymlPath = path.join(absolute, "inference.yml");
      await assertReadable(onnxPath, "ONNX model");
      return {
        onnxPath,
        ymlPath: fsSync.existsSync(ymlPath) ? ymlPath : null,
      };
    }
    if (absolute.endsWith(".onnx")) {
      return {
        onnxPath: absolute,
        ymlPath: null,
      };
    }
    throw new Error("--model local path must be a directory or an .onnx file");
  }

  const targetDir = path.join(cacheDir, sanitizeModelId(modelRef));
  const onnxPath = path.join(targetDir, "inference.onnx");
  const ymlPath = path.join(targetDir, "inference.yml");
  await fs.mkdir(targetDir, { recursive: true });
  await downloadIfMissing(hfResolveUrl(modelRef, "inference.onnx"), onnxPath);
  await downloadIfMissing(hfRawUrl(modelRef, "inference.yml"), ymlPath);
  return { onnxPath, ymlPath };
}

async function assertReadable(filePath, label) {
  try {
    await fs.access(filePath, fsSync.constants.R_OK);
  } catch {
    throw new Error(`${label} not readable: ${filePath}`);
  }
}

function sanitizeModelId(modelId) {
  return modelId.replaceAll("/", "__").replaceAll(/[^A-Za-z0-9_.-]/g, "_");
}

function hfResolveUrl(modelId, filename) {
  return `${HF_BASE_URL}/${modelId}/resolve/main/${filename}`;
}

function hfRawUrl(modelId, filename) {
  return `${HF_BASE_URL}/${modelId}/raw/main/${filename}`;
}

async function downloadIfMissing(url, dest) {
  if (fsSync.existsSync(dest) && fsSync.statSync(dest).size > 0) {
    return;
  }
  console.log(`Downloading ${url}`);
  const response = await undiciFetch(url, {
    dispatcher: getProxyAgent(url),
    redirect: "follow",
  });
  if (!response.ok) {
    throw new Error(`Download failed ${response.status} ${response.statusText}: ${url}`);
  }
  const temp = `${dest}.tmp`;
  const file = fsSync.createWriteStream(temp);
  await new Promise((resolve, reject) => {
    response.body.pipeTo(
      new WritableStream({
        write(chunk) {
          file.write(Buffer.from(chunk));
        },
        close() {
          file.end(resolve);
        },
        abort(error) {
          file.destroy(error);
          reject(error);
        },
      }),
    ).catch(reject);
  });
  await fs.rename(temp, dest);
}

function getProxyAgent(url) {
  const proxy =
    url.startsWith("https:")
      ? process.env.HTTPS_PROXY ?? process.env.https_proxy ?? process.env.HTTP_PROXY ?? process.env.http_proxy
      : process.env.HTTP_PROXY ?? process.env.http_proxy;
  return proxy ? new ProxyAgent(proxy) : undefined;
}

async function checkTransformersPipeline(modelRef, modelFiles) {
  const pipelineModelRef = fsSync.existsSync(path.resolve(modelRef)) ? DEFAULT_HF_MODEL : modelRef;
  const result = {
    attempted: true,
    task: "image-to-text",
    model: pipelineModelRef,
    ok: false,
    repositoryShape: {
      hasInferenceOnnx: true,
      hasInferenceYml: Boolean(modelFiles.ymlPath),
      hasTransformersConfig: false,
      hasOnnxModelDir: false,
    },
    error: "",
  };
  const localDir = path.dirname(modelFiles.onnxPath);
  result.repositoryShape.hasTransformersConfig = fsSync.existsSync(path.join(localDir, "config.json"));
  result.repositoryShape.hasOnnxModelDir = fsSync.existsSync(path.join(localDir, "onnx", "model.onnx"));

  try {
    const { pipeline } = await import("@huggingface/transformers");
    await pipeline("image-to-text", pipelineModelRef);
    result.ok = true;
    return result;
  } catch (error) {
    result.error = `${normalizeError(error)}. Repository probe: this model cache contains inference.onnx/inference.yml, but not transformers.js config.json + onnx/model.onnx pipeline assets.`;
    return result;
  }
}

function normalizeError(error) {
  const message = error instanceof Error ? error.message : String(error);
  return message.replaceAll(/\s+/g, " ").trim();
}

async function preprocessImage(imagePath) {
  const source = sharp(imagePath);
  const metadata = await source.metadata();
  if (!metadata.width || !metadata.height) {
    throw new Error(`Unable to read image dimensions: ${imagePath}`);
  }

  const srcW = metadata.width;
  const srcH = metadata.height;
  const resized = resizeForDet(srcW, srcH);
  const { data } = await sharp(imagePath)
    .resize(resized.width, resized.height, { fit: "fill" })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const chw = new Float32Array(3 * resized.width * resized.height);
  const planeSize = resized.width * resized.height;
  for (let y = 0; y < resized.height; y += 1) {
    for (let x = 0; x < resized.width; x += 1) {
      const srcIndex = (y * resized.width + x) * 3;
      const dstIndex = y * resized.width + x;
      const rgb = [
        data[srcIndex] / 255,
        data[srcIndex + 1] / 255,
        data[srcIndex + 2] / 255,
      ];
      for (let c = 0; c < 3; c += 1) {
        chw[c * planeSize + dstIndex] = (rgb[c] - NORMALIZE.mean[c]) / NORMALIZE.std[c];
      }
    }
  }

  return {
    data: chw,
    width: resized.width,
    height: resized.height,
    shape: {
      srcH,
      srcW,
      ratioH: resized.height / srcH,
      ratioW: resized.width / srcW,
    },
  };
}

function resizeForDet(width, height) {
  let resizeW = width;
  let resizeH = height;

  if (DET_LIMIT_TYPE === "max") {
    const ratio = Math.max(resizeH, resizeW);
    if (ratio > DET_LIMIT_SIDE_LEN) {
      if (resizeH > resizeW) {
        resizeH = DET_LIMIT_SIDE_LEN;
        resizeW = Math.round((width * DET_LIMIT_SIDE_LEN) / height);
      } else {
        resizeW = DET_LIMIT_SIDE_LEN;
        resizeH = Math.round((height * DET_LIMIT_SIDE_LEN) / width);
      }
    }
  }

  resizeH = Math.max(32, Math.round(resizeH / 32) * 32);
  resizeW = Math.max(32, Math.round(resizeW / 32) * 32);
  return { width: resizeW, height: resizeH };
}

function postprocessDetection(outputTensor, shape) {
  const dims = outputTensor.dims;
  if (dims.length !== 4) {
    throw new Error(`Expected 4D model output, got shape ${JSON.stringify(dims)}`);
  }
  const predH = dims[2];
  const predW = dims[3];
  const pred = outputTensor.data;
  const bitmap = new Uint8Array(predH * predW);
  for (let i = 0; i < bitmap.length; i += 1) {
    bitmap[i] = pred[i] > POSTPROCESS_DEFAULTS.thresh ? 1 : 0;
  }

  const boxes = [];
  const components = findConnectedComponents(bitmap, pred, predW, predH);
  components.sort((a, b) => b.pixelCount - a.pixelCount);

  for (const component of components.slice(0, POSTPROCESS_DEFAULTS.maxCandidates)) {
    const width = component.xmax - component.xmin + 1;
    const height = component.ymax - component.ymin + 1;
    if (Math.min(width, height) < POSTPROCESS_DEFAULTS.minSize) {
      continue;
    }
    const score = component.scoreSum / component.pixelCount;
    if (score < POSTPROCESS_DEFAULTS.boxThresh) {
      continue;
    }

    const baseBox = [
      [component.xmin, component.ymin],
      [component.xmax, component.ymin],
      [component.xmax, component.ymax],
      [component.xmin, component.ymax],
    ];
    const expanded = unclipAabb(baseBox, POSTPROCESS_DEFAULTS.unclipRatio);
    const expandedWidth =
      Math.max(...expanded.map((point) => point[0])) -
      Math.min(...expanded.map((point) => point[0]));
    const expandedHeight =
      Math.max(...expanded.map((point) => point[1])) -
      Math.min(...expanded.map((point) => point[1]));
    if (Math.min(expandedWidth, expandedHeight) < POSTPROCESS_DEFAULTS.minSize + 2) {
      continue;
    }

    const points = expanded.map(([x, y]) => [
      clamp(Math.round((x / predW) * shape.srcW), 0, shape.srcW),
      clamp(Math.round((y / predH) * shape.srcH), 0, shape.srcH),
    ]);
    boxes.push({ points, score: round(score, 6) });
  }

  boxes.sort((a, b) => {
    const ay = average(a.points.map((point) => point[1]));
    const by = average(b.points.map((point) => point[1]));
    if (Math.abs(ay - by) > 10) {
      return ay - by;
    }
    return average(a.points.map((point) => point[0])) - average(b.points.map((point) => point[0]));
  });
  return boxes;
}

function findConnectedComponents(bitmap, pred, width, height) {
  const visited = new Uint8Array(bitmap.length);
  const components = [];
  const queue = new Int32Array(bitmap.length);
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
    queue[tail] = start;
    tail += 1;
    visited[start] = 1;

    let xmin = width;
    let xmax = 0;
    let ymin = height;
    let ymax = 0;
    let pixelCount = 0;
    let scoreSum = 0;

    while (head < tail) {
      const current = queue[head];
      head += 1;
      const x = current % width;
      const y = Math.floor(current / width);
      xmin = Math.min(xmin, x);
      xmax = Math.max(xmax, x);
      ymin = Math.min(ymin, y);
      ymax = Math.max(ymax, y);
      pixelCount += 1;
      scoreSum += pred[current];

      for (const [dx, dy] of directions) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= width || ny >= height) {
          continue;
        }
        const next = ny * width + nx;
        if (bitmap[next] === 0 || visited[next] !== 0) {
          continue;
        }
        visited[next] = 1;
        queue[tail] = next;
        tail += 1;
      }
    }

    components.push({ xmin, xmax, ymin, ymax, pixelCount, scoreSum });
  }

  return components;
}

function unclipAabb(box, unclipRatio) {
  const area = polygonArea(box);
  const length = polygonLength(box);
  if (area <= 0 || length <= 0) {
    return box;
  }
  const distance = (area * unclipRatio) / length;
  const xs = box.map((point) => point[0]);
  const ys = box.map((point) => point[1]);
  const xmin = Math.min(...xs) - distance;
  const xmax = Math.max(...xs) + distance;
  const ymin = Math.min(...ys) - distance;
  const ymax = Math.max(...ys) + distance;
  return [
    [xmin, ymin],
    [xmax, ymin],
    [xmax, ymax],
    [xmin, ymax],
  ];
}

function polygonArea(points) {
  let area = 0;
  for (let i = 0; i < points.length; i += 1) {
    const [x1, y1] = points[i];
    const [x2, y2] = points[(i + 1) % points.length];
    area += x1 * y2 - x2 * y1;
  }
  return Math.abs(area) / 2;
}

function polygonLength(points) {
  let length = 0;
  for (let i = 0; i < points.length; i += 1) {
    const [x1, y1] = points[i];
    const [x2, y2] = points[(i + 1) % points.length];
    length += Math.hypot(x2 - x1, y2 - y1);
  }
  return length;
}

async function drawDebugImage(imagePath, outputPath, boxes) {
  const metadata = await sharp(imagePath).metadata();
  if (!metadata.width || !metadata.height) {
    throw new Error(`Unable to read image dimensions: ${imagePath}`);
  }
  const overlays = boxes.flatMap((box, index) => {
    const color = index % 2 === 0 ? "#00D084" : "#FF4D4F";
    return [
      `<polygon points="${box.points.map(([x, y]) => `${x},${y}`).join(" ")}" fill="none" stroke="${color}" stroke-width="3"/>`,
      `<text x="${box.points[0][0]}" y="${Math.max(12, box.points[0][1] - 4)}" fill="${color}" font-size="18" font-family="sans-serif">${index + 1}</text>`,
    ];
  });
  const svg = `<svg width="${metadata.width}" height="${metadata.height}" viewBox="0 0 ${metadata.width} ${metadata.height}" xmlns="http://www.w3.org/2000/svg">${overlays.join("")}</svg>`;
  await fs.mkdir(path.dirname(path.resolve(outputPath)), { recursive: true });
  await sharp(imagePath).composite([{ input: Buffer.from(svg), top: 0, left: 0 }]).toFile(outputPath);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function average(values) {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function round(value, places) {
  const scale = 10 ** places;
  return Math.round(value * scale) / scale;
}
