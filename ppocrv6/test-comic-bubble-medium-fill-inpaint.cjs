#!/usr/bin/env node
const fs = require("node:fs/promises");
const path = require("node:path");

const ClipperLib = require("clipper-lib");
const cv = require("@techstark/opencv-js");
const ort = require("onnxruntime-web");
const sharp = require("sharp");
const yaml = require("js-yaml");

const SCRIPT_DIR = __dirname;
const ASSETS_DIR = path.join(SCRIPT_DIR, "assets");

const COMIC_MODEL_DIR = path.join(SCRIPT_DIR, "model", "comic-bubble");
const COMIC_MODEL_ONNX_PATH = path.join(COMIC_MODEL_DIR, "detector-v4-s_int8.onnx");
const COMIC_MODEL_CONFIG_PATH = path.join(COMIC_MODEL_DIR, "config.json");
const COMIC_PREPROCESSOR_CONFIG_PATH = path.join(COMIC_MODEL_DIR, "preprocessor_config.json");

const DET_MODEL_DIR = path.join(SCRIPT_DIR, "model", "medium-det");
const DET_MODEL_ONNX_PATH = path.join(DET_MODEL_DIR, "inference.onnx");
const DET_MODEL_YML_PATH = path.join(DET_MODEL_DIR, "inference.yml");

const DEFAULT_INPUT_IMAGE_PATHS = [
  path.join(ASSETS_DIR, "cct.png"),
  path.join(ASSETS_DIR, "cctt.png"),
];

const INPUT_IMAGE_PATHS = (process.env.INPUT_IMAGES ?? "")
  .split(",")
  .map((entry) => entry.trim())
  .filter(Boolean)
  .map((entry) => path.resolve(SCRIPT_DIR, entry));

const COMIC_SCORE_THRESHOLD = readNumberEnv("COMIC_BUBBLE_THRESHOLD", 0.35);
const CONTEXT_LABELS = readStringListEnv("COMIC_FILL_CONTEXT_LABELS", ["bubble"]);
const TEXT_LABELS = readStringListEnv("COMIC_FILL_TEXT_LABELS", ["text_bubble", "text_free"]);
const MASK_PADDING = readNumberEnv("COMIC_FILL_MASK_PADDING", 2);
const RING_PADDING = readNumberEnv("COMIC_FILL_RING_PADDING", 8);
const MERGE_GAP = readNumberEnv("COMIC_FILL_MERGE_GAP", 6);
const MIN_RING_PIXELS = readNumberEnv("COMIC_FILL_MIN_RING_PIXELS", 24);
const FILL_LIMIT = readNumberEnv("COMIC_FILL_LIMIT", 0);

const MIN_SIZE = 3;
const CLIPPER_SCALE = 1024;

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
});

async function main() {
  if (process.argv.includes("-h") || process.argv.includes("--help")) {
    printHelp();
    return;
  }

  const inputImagePaths = INPUT_IMAGE_PATHS.length > 0 ? INPUT_IMAGE_PATHS : DEFAULT_INPUT_IMAGE_PATHS;

  console.time("opencv");
  await waitForOpenCv();
  console.timeEnd("opencv");

  await Promise.all([
    assertFile(COMIC_MODEL_ONNX_PATH, "make comic-bubble"),
    assertFile(DET_MODEL_ONNX_PATH, "make medium-det"),
  ]);

  const [comicModelConfig, comicPreprocessorConfig, detConfig] = await Promise.all([
    readJson(COMIC_MODEL_CONFIG_PATH),
    readJson(COMIC_PREPROCESSOR_CONFIG_PATH),
    readDetectionConfig(DET_MODEL_YML_PATH),
  ]);

  const comicImageSize = normalizeComicImageSize(comicPreprocessorConfig);
  const comicId2Label = normalizeId2Label(comicModelConfig.id2label);

  console.time("comic-session");
  const comicSession = await createSession(COMIC_MODEL_ONNX_PATH);
  console.timeEnd("comic-session");

  console.time("medium-det-session");
  const detSession = await createSession(DET_MODEL_ONNX_PATH);
  console.timeEnd("medium-det-session");

  const comicInputName = comicSession.inputNames.includes("images")
    ? "images"
    : comicSession.inputNames[0];
  const comicTargetSizeInputName = comicSession.inputNames.includes("orig_target_sizes")
    ? "orig_target_sizes"
    : comicSession.inputNames[1];
  const detInputName = detSession.inputNames[0] ?? "x";
  const detOutputName = detSession.outputNames[0] ?? "fetch_name_0";

  if (!comicInputName || !comicTargetSizeInputName) {
    throw new Error(`Unexpected comic-bubble model inputs: ${comicSession.inputNames.join(", ")}`);
  }

  for (const imagePath of inputImagePaths) {
    await runImage({
      imagePath,
      comicSession,
      comicInputName,
      comicTargetSizeInputName,
      comicImageSize,
      comicId2Label,
      comicPreprocessorConfig,
      detSession,
      detInputName,
      detOutputName,
      detConfig,
    });
  }
}

function printHelp() {
  console.log(`Usage:
  npm run inpaint:comic:fill
  INPUT_IMAGES=assets/cct.png npm run inpaint:comic:fill

Models:
  make comic-bubble
  make medium-det

Environment:
  INPUT_IMAGES                 Comma-separated image paths, relative to ppocrv6/ or absolute.
  COMIC_BUBBLE_THRESHOLD       Comic detector score threshold. Default: ${COMIC_SCORE_THRESHOLD}
  COMIC_FILL_CONTEXT_LABELS    Primary comic labels used for assignment. Default: ${CONTEXT_LABELS.join(",")}
  COMIC_FILL_TEXT_LABELS       Text labels always handled before bubble fallback. Default: ${TEXT_LABELS.join(",")}
  COMIC_FILL_MASK_PADDING      Pixels dilated around each text polygon mask. Default: ${MASK_PADDING}
  COMIC_FILL_RING_PADDING      Pixels used for the merged text outer sampling ring. Default: ${RING_PADDING}
  COMIC_FILL_MERGE_GAP         Max bbox gap for merging text boxes in the same region. Default: ${MERGE_GAP}
  COMIC_FILL_MIN_RING_PIXELS   Minimum sampled pixels before fallback. Default: ${MIN_RING_PIXELS}
  COMIC_FILL_LIMIT             Max fill groups processed; 0 means all. Default: ${FILL_LIMIT}
`);
}

async function createSession(modelPath) {
  const modelBytes = await fs.readFile(modelPath);
  return ort.InferenceSession.create(modelBytes, {
    executionProviders: ["wasm"],
  });
}

async function runImage({
  imagePath,
  comicSession,
  comicInputName,
  comicTargetSizeInputName,
  comicImageSize,
  comicId2Label,
  comicPreprocessorConfig,
  detSession,
  detInputName,
  detOutputName,
  detConfig,
}) {
  const imageBaseName = path.basename(imagePath, path.extname(imagePath));
  const outputJsonPath = path.join(ASSETS_DIR, `${imageBaseName}.comic-medium-fill-inpaint.json`);
  const outputImagePath = path.join(ASSETS_DIR, `${imageBaseName}.comic-medium-fill-clean.png`);
  const maskImagePath = path.join(ASSETS_DIR, `${imageBaseName}.comic-medium-fill-mask.png`);
  const debugImagePath = path.join(ASSETS_DIR, `${imageBaseName}.comic-medium-fill-debug.png`);

  console.log(`\n${path.relative(SCRIPT_DIR, imagePath)}`);

  const metadata = await sharp(imagePath).metadata();
  if (!metadata.width || !metadata.height) {
    throw new Error(`Unable to read image dimensions: ${imagePath}`);
  }
  const imageWidth = metadata.width;
  const imageHeight = metadata.height;

  const { data: imageData } = await sharp(imagePath)
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const outputData = Buffer.from(imageData);
  const pageMask = Buffer.alloc(imageWidth * imageHeight, 0);

  console.time(`${imageBaseName}:comic-preprocess`);
  const comicImage = await preprocessComicImage(imagePath, comicImageSize, comicPreprocessorConfig);
  console.timeEnd(`${imageBaseName}:comic-preprocess`);

  const comicImageTensor = new ort.Tensor("float32", comicImage.tensorData, [
    1,
    3,
    comicImageSize.height,
    comicImageSize.width,
  ]);
  const comicTargetSizeTensor = new ort.Tensor("int64", BigInt64Array.from([
    BigInt(comicImage.originalWidth),
    BigInt(comicImage.originalHeight),
  ]), [1, 2]);

  console.time(`${imageBaseName}:comic-inference`);
  const comicOutputs = await comicSession.run({
    [comicInputName]: comicImageTensor,
    [comicTargetSizeInputName]: comicTargetSizeTensor,
  });
  console.timeEnd(`${imageBaseName}:comic-inference`);

  const comicDetections = decodeComicDetections(
    comicOutputs,
    comicId2Label,
    imageWidth,
    imageHeight,
  )
    .filter((detection) => detection.score >= COMIC_SCORE_THRESHOLD)
    .map((detection, index) => ({
      index,
      ...detection,
      bbox: detection.bbox.map((value) => round(value, 3)),
    }));

  console.time(`${imageBaseName}:medium-det-preprocess`);
  const detImage = await preprocessDetectionImage(imagePath, detConfig.preprocessing);
  console.timeEnd(`${imageBaseName}:medium-det-preprocess`);

  const detInputTensor = new ort.Tensor("float32", detImage.tensorData, [
    1,
    3,
    detImage.height,
    detImage.width,
  ]);

  console.time(`${imageBaseName}:medium-det-inference`);
  const detOutputs = await detSession.run({ [detInputName]: detInputTensor });
  console.timeEnd(`${imageBaseName}:medium-det-inference`);

  const detOutputTensor = detOutputs[detOutputName] ?? detOutputs[detSession.outputNames[0]];
  if (!detOutputTensor) {
    throw new Error(`Detection model output not found. Available outputs: ${Object.keys(detOutputs).join(", ")}`);
  }

  console.time(`${imageBaseName}:medium-det-postprocess`);
  const mediumDetBoxes = detectBoxes(
    detOutputTensor,
    detImage.originalWidth,
    detImage.originalHeight,
    detConfig.postprocess,
  )
    .map((box, index) => normalizeDetectedBox(box, index, imageWidth, imageHeight))
    .sort(compareDetectedBoxReadingOrder)
    .map((box, index) => ({
      ...box,
      index,
    }));
  console.timeEnd(`${imageBaseName}:medium-det-postprocess`);

  const { assignments, assignedBoxes, skippedBoxes } = assignMediumBoxes({
    mediumDetBoxes,
    comicDetections,
  });
  const fillGroups = buildFillGroups({
    assignedBoxes,
    imageWidth,
    imageHeight,
  });
  applyFillLimit(fillGroups, FILL_LIMIT);

  const pageTextMask = Buffer.alloc(imageWidth * imageHeight, 0);

  for (const group of fillGroups.filter((item) => !item.skipped)) {
    const maskBuffer = createGroupMaskBuffer({
      group,
      imageWidth,
      imageHeight,
      maskPadding: MASK_PADDING,
    });
    group.maskBuffer = maskBuffer;
    group.maskBox = maskBufferBbox(maskBuffer, imageWidth, imageHeight);
    if (!group.maskBox) {
      group.skipped = true;
      group.skipReason = "empty rasterized mask";
      continue;
    }
    orMaskInto(pageTextMask, maskBuffer);
    orMaskInto(pageMask, maskBuffer);
  }

  const activeGroups = fillGroups.filter((group) => !group.skipped);
  for (const group of activeGroups) {
    const sample = sampleFillColor({
      imageData,
      imageWidth,
      imageHeight,
      groupMask: group.maskBuffer,
      pageTextMask,
      regionBox: group.region.bbox,
      ringPadding: RING_PADDING,
      minPixels: MIN_RING_PIXELS,
    });
    group.fillColor = sample.color;
    group.fillSource = sample.source;
    group.sampleCount = sample.sampleCount;
    group.ringPaddingUsed = sample.ringPaddingUsed;
    fillGroupInImage({
      outputData,
      imageWidth,
      groupMask: group.maskBuffer,
      color: sample.color,
    });
  }

  await sharp(outputData, {
    raw: {
      width: imageWidth,
      height: imageHeight,
      channels: 3,
    },
  }).png().toFile(outputImagePath);

  await sharp(pageMask, {
    raw: {
      width: imageWidth,
      height: imageHeight,
      channels: 1,
    },
  }).png().toFile(maskImagePath);

  await drawDebugImage({
    imagePath,
    outputPath: debugImagePath,
    imageWidth,
    imageHeight,
    pageMask,
    comicDetections,
    mediumDetBoxes,
    fillGroups,
  });

  const result = {
    image: path.resolve(imagePath),
    outputs: {
      cleanedImage: outputImagePath,
      mask: maskImagePath,
      debug: debugImagePath,
    },
    models: {
      comicBubble: {
        repository: "ogkalu/comic-text-and-bubble-detector",
        variant: "detector-v4-s_int8.onnx",
        onnx: COMIC_MODEL_ONNX_PATH,
        config: COMIC_MODEL_CONFIG_PATH,
        preprocessor: COMIC_PREPROCESSOR_CONFIG_PATH,
      },
      textDetection: {
        repository: "PaddlePaddle/PP-OCRv6_medium_det_onnx",
        onnx: DET_MODEL_ONNX_PATH,
        yml: DET_MODEL_YML_PATH,
      },
    },
    runtime: {
      engine: "onnxruntime-web",
      backend: "wasm",
      comicBubble: {
        inputNames: comicSession.inputNames,
        outputNames: comicSession.outputNames,
        inputShape: [1, 3, comicImageSize.height, comicImageSize.width],
      },
      textDetection: {
        inputNames: detSession.inputNames,
        outputNames: detSession.outputNames,
        inputShape: [1, 3, detImage.height, detImage.width],
        outputShape: detOutputTensor.dims,
      },
    },
    preprocessing: {
      comicBubble: {
        originalWidth: comicImage.originalWidth,
        originalHeight: comicImage.originalHeight,
        resizedWidth: comicImageSize.width,
        resizedHeight: comicImageSize.height,
        doRescale: Boolean(comicPreprocessorConfig.do_rescale),
        rescaleFactor: Number(comicPreprocessorConfig.rescale_factor ?? 1 / 255),
        doNormalize: Boolean(comicPreprocessorConfig.do_normalize),
      },
      textDetection: {
        originalWidth: detImage.originalWidth,
        originalHeight: detImage.originalHeight,
        resizedWidth: detImage.width,
        resizedHeight: detImage.height,
        ratioH: detImage.ratioH,
        ratioW: detImage.ratioW,
        imgMode: detConfig.preprocessing.imgMode,
        scale: detConfig.preprocessing.scale,
        mean: detConfig.preprocessing.mean,
        std: detConfig.preprocessing.std,
        detResizeForTest: detConfig.preprocessing.detResizeForTest,
      },
    },
    settings: {
      comicScoreThreshold: COMIC_SCORE_THRESHOLD,
      contextLabels: CONTEXT_LABELS,
      textLabels: TEXT_LABELS,
      maskPadding: MASK_PADDING,
      ringPadding: RING_PADDING,
      mergeGap: MERGE_GAP,
      minRingPixels: MIN_RING_PIXELS,
      fillLimit: FILL_LIMIT,
      inpaintMode: "outer-ring-median-fill",
    },
    counts: {
      comicDetections: countByLabel(comicDetections),
      mediumDetBoxes: mediumDetBoxes.length,
      assignedBoxes: assignedBoxes.length,
      skippedBoxes: skippedBoxes.length,
      fillGroups: fillGroups.length,
      processedFillGroups: activeGroups.length,
      skippedFillGroups: fillGroups.filter((group) => group.skipped).length,
    },
    comicDetections,
    mediumDetBoxes,
    assignments,
    skippedBoxes,
    fillGroups: fillGroups.map(serializeFillGroup),
  };

  await fs.writeFile(outputJsonPath, `${JSON.stringify(result, null, 2)}\n`, "utf8");

  console.log(`comic detections: ${comicDetections.length}`, result.counts.comicDetections);
  console.log(`medium-det boxes: ${mediumDetBoxes.length}`);
  console.log(`assigned boxes: ${assignedBoxes.length}; skipped boxes: ${skippedBoxes.length}`);
  console.log(`fill groups: ${fillGroups.length}; processed: ${activeGroups.length}`);
  console.log(`clean: ${path.relative(SCRIPT_DIR, outputImagePath)}`);
  console.log(`mask: ${path.relative(SCRIPT_DIR, maskImagePath)}`);
  console.log(`debug: ${path.relative(SCRIPT_DIR, debugImagePath)}`);
  console.log(`json: ${path.relative(SCRIPT_DIR, outputJsonPath)}`);
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

async function preprocessComicImage(imagePath, imageSize, preprocessorConfig) {
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

async function preprocessDetectionImage(imageSource, preprocessing) {
  const metadata = await sharp(imageSource).metadata();
  if (!metadata.width || !metadata.height) {
    throw new Error("Unable to read detection image dimensions.");
  }

  const size = resizeForTest(metadata.width, metadata.height, preprocessing.detResizeForTest);
  const { data } = await sharp(imageSource)
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

function decodeComicDetections(outputs, id2label, imageWidth, imageHeight) {
  const labelsTensor = outputs.labels;
  const boxesTensor = outputs.boxes;
  const scoresTensor = outputs.scores;

  if (!labelsTensor || !boxesTensor || !scoresTensor) {
    throw new Error(`Unexpected comic-bubble model outputs: ${Object.keys(outputs).join(", ")}`);
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
    detections.push({
      labelId,
      label: id2label[labelId] ?? String(labelId),
      score: round(Number(scoresTensor.data[index]), 6),
      bbox: [
        clamp(bbox[0], 0, imageWidth),
        clamp(bbox[1], 0, imageHeight),
        clamp(bbox[2], 0, imageWidth),
        clamp(bbox[3], 0, imageHeight),
      ],
    });
  }

  return detections.sort((a, b) => b.score - a.score);
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
            score: round(score, 6),
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

function normalizeDetectedBox(box, index, imageWidth, imageHeight) {
  const points = box.points.map(([x, y]) => [
    clamp(Number(x), 0, imageWidth),
    clamp(Number(y), 0, imageHeight),
  ]);
  const bbox = pointsToBbox(points, imageWidth, imageHeight);
  return {
    index,
    score: round(box.score, 6),
    pointsInImage: points.map((point) => point.map((value) => round(value, 3))),
    bboxInImage: bbox,
    center: bboxCenter(bbox).map((value) => round(value, 3)),
  };
}

function assignMediumBoxes({ mediumDetBoxes, comicDetections }) {
  const contextLabelSet = new Set(CONTEXT_LABELS);
  const textLabelSet = new Set(TEXT_LABELS);
  const contextDetections = comicDetections.filter((detection) => contextLabelSet.has(detection.label));
  const textDetections = comicDetections.filter((detection) => textLabelSet.has(detection.label));
  const assignments = [];
  const assignedBoxes = [];
  const skippedBoxes = [];
  const assignedTextDetectionIndexes = new Set();

  for (const box of mediumDetBoxes) {
    const text = findSmallestContainingDetection(box.center, textDetections);
    const context = text ? null : findSmallestContainingDetection(box.center, contextDetections);
    const detection = text ?? context;
    if (!detection) {
      const skipped = {
        boxIndex: box.index,
        reason: "center outside comic regions",
      };
      skippedBoxes.push(skipped);
      assignments.push({
        boxIndex: box.index,
        assigned: false,
        skipReason: skipped.reason,
      });
      continue;
    }

    const regionSource = text ? "text" : "context";
    if (text) {
      assignedTextDetectionIndexes.add(text.index);
    }
    const region = {
      source: regionSource,
      detectionIndex: detection.index,
      label: detection.label,
      score: detection.score,
      bbox: integerBox(detection.bbox, Number.MAX_SAFE_INTEGER, Number.MAX_SAFE_INTEGER),
    };
    const assigned = {
      ...box,
      region,
    };
    assignedBoxes.push(assigned);
    assignments.push({
      boxIndex: box.index,
      assigned: true,
      region,
    });
  }

  for (const detection of textDetections) {
    if (assignedTextDetectionIndexes.has(detection.index)) {
      continue;
    }
    const bbox = integerBox(detection.bbox, Number.MAX_SAFE_INTEGER, Number.MAX_SAFE_INTEGER);
    const fallbackBox = {
      index: `text-fallback-${detection.index}`,
      source: "comic-text-detection-fallback",
      score: detection.score,
      pointsInImage: boxToPoints(bbox),
      bboxInImage: bbox,
      center: bboxCenter(bbox).map((value) => round(value, 3)),
      region: {
        source: "text",
        detectionIndex: detection.index,
        label: detection.label,
        score: detection.score,
        bbox,
      },
    };
    assignedBoxes.push(fallbackBox);
    assignments.push({
      boxIndex: fallbackBox.index,
      assigned: true,
      fallback: true,
      region: fallbackBox.region,
    });
  }

  return {
    assignments,
    assignedBoxes,
    skippedBoxes,
  };
}

function findSmallestContainingDetection(point, detections) {
  let best = null;
  let bestArea = Infinity;
  for (const detection of detections) {
    if (!pointInBox(point, detection.bbox)) {
      continue;
    }
    const area = bboxArea(detection.bbox);
    if (area < bestArea) {
      best = detection;
      bestArea = area;
    }
  }
  return best;
}

function buildFillGroups({ assignedBoxes, imageWidth, imageHeight }) {
  const byRegion = new Map();
  for (const box of assignedBoxes) {
    const key = `${box.region.source}:${box.region.detectionIndex}`;
    if (!byRegion.has(key)) {
      byRegion.set(key, []);
    }
    byRegion.get(key).push(box);
  }

  const groups = [];
  for (const boxes of byRegion.values()) {
    for (const component of connectedBoxComponents(boxes, MERGE_GAP)) {
      const maskSourceBox = component.reduce((box, item) => unionBoxes(box, item.bboxInImage), component[0].bboxInImage);
      groups.push({
        index: groups.length,
        region: component[0].region,
        boxIndexes: component.map((box) => box.index),
        boxes: component.map(serializeMediumDetBox),
        maskSourceBox: integerBox(maskSourceBox, imageWidth, imageHeight),
        maskBox: null,
        fillColor: null,
        fillSource: null,
        sampleCount: 0,
        ringPaddingUsed: 0,
        skipped: false,
        skipReason: null,
      });
    }
  }

  return groups
    .sort(compareFillGroupReadingOrder)
    .map((group, index) => ({
      ...group,
      index,
    }));
}

function connectedBoxComponents(boxes, maxGap) {
  const remaining = new Set(boxes.map((_, index) => index));
  const components = [];
  while (remaining.size > 0) {
    const [first] = remaining;
    remaining.delete(first);
    const queue = [first];
    const componentIndexes = [first];
    for (let cursor = 0; cursor < queue.length; cursor += 1) {
      const current = queue[cursor];
      for (const candidate of [...remaining]) {
        if (bboxDistance(boxes[current].bboxInImage, boxes[candidate].bboxInImage) <= maxGap) {
          remaining.delete(candidate);
          queue.push(candidate);
          componentIndexes.push(candidate);
        }
      }
    }
    components.push(componentIndexes.map((index) => boxes[index]).sort(compareDetectedBoxReadingOrder));
  }
  return components;
}

function applyFillLimit(fillGroups, limit) {
  if (limit <= 0) {
    return;
  }
  for (const group of fillGroups.slice(limit)) {
    group.skipped = true;
    group.skipReason = "fill limit";
  }
}

function createGroupMaskBuffer({ group, imageWidth, imageHeight, maskPadding }) {
  let mask = cv.Mat.zeros(imageHeight, imageWidth, cv.CV_8UC1);
  try {
    for (const box of group.boxes) {
      fillPolygonOnMask(mask, box.pointsInImage, 255);
    }

    if (maskPadding > 0) {
      const kernelSize = Math.max(1, Math.round(maskPadding) * 2 + 1);
      const kernel = cv.Mat.ones(kernelSize, kernelSize, cv.CV_8UC1);
      const dilated = new cv.Mat();
      try {
        cv.dilate(mask, dilated, kernel);
      } finally {
        kernel.delete();
        mask.delete();
      }
      mask = dilated;
    }

    restrictMaskToBox(mask.data, imageWidth, imageHeight, group.region.bbox);
    return Buffer.from(mask.data);
  } finally {
    mask.delete();
  }
}

function fillPolygonOnMask(mask, points, value) {
  const polygon = pointsArrayToMat(points);
  const polygons = new cv.MatVector();
  try {
    polygons.push_back(polygon);
    cv.fillPoly(mask, polygons, new cv.Scalar(value));
  } finally {
    polygon.delete();
    polygons.delete();
  }
}

function pointsArrayToMat(points) {
  const mat = new cv.Mat(points.length, 1, cv.CV_32SC2);
  for (let index = 0; index < points.length; index += 1) {
    mat.data32S[index * 2] = Math.round(points[index][0]);
    mat.data32S[index * 2 + 1] = Math.round(points[index][1]);
  }
  return mat;
}

function restrictMaskToBox(mask, width, height, box) {
  const [x1, y1, x2, y2] = integerBox(box, width, height);
  for (let y = 0; y < height; y += 1) {
    const row = y * width;
    if (y < y1 || y >= y2) {
      mask.fill(0, row, row + width);
      continue;
    }
    mask.fill(0, row, row + x1);
    mask.fill(0, row + x2, row + width);
  }
}

function orMaskInto(target, source) {
  for (let index = 0; index < target.length; index += 1) {
    if (source[index] > 0) {
      target[index] = 255;
    }
  }
}

function sampleFillColor({
  imageData,
  imageWidth,
  imageHeight,
  groupMask,
  pageTextMask,
  regionBox,
  ringPadding,
  minPixels,
}) {
  const first = collectRingSamples({
    imageData,
    imageWidth,
    imageHeight,
    groupMask,
    pageTextMask,
    regionBox,
    ringPadding,
  });
  if (first.count >= minPixels) {
    return {
      color: medianSampleColor(first),
      source: "outer-ring",
      sampleCount: first.count,
      ringPaddingUsed: ringPadding,
    };
  }

  const expandedPadding = Math.max(ringPadding + 1, ringPadding * 2);
  const expanded = collectRingSamples({
    imageData,
    imageWidth,
    imageHeight,
    groupMask,
    pageTextMask,
    regionBox,
    ringPadding: expandedPadding,
  });
  if (expanded.count >= minPixels) {
    return {
      color: medianSampleColor(expanded),
      source: "expanded-outer-ring",
      sampleCount: expanded.count,
      ringPaddingUsed: expandedPadding,
    };
  }

  const region = collectRegionSamples({
    imageData,
    imageWidth,
    imageHeight,
    pageTextMask,
    regionBox,
  });
  if (region.count > 0) {
    return {
      color: medianSampleColor(region),
      source: "region-non-mask",
      sampleCount: region.count,
      ringPaddingUsed: 0,
    };
  }

  return {
    color: [255, 255, 255],
    source: "fallback-white",
    sampleCount: 0,
    ringPaddingUsed: 0,
  };
}

function collectRingSamples({
  imageData,
  imageWidth,
  imageHeight,
  groupMask,
  pageTextMask,
  regionBox,
  ringPadding,
}) {
  const ringMask = dilateMaskBuffer({
    maskBuffer: groupMask,
    width: imageWidth,
    height: imageHeight,
    padding: ringPadding,
  });
  const [x1, y1, x2, y2] = integerBox(regionBox, imageWidth, imageHeight);
  const samples = emptySamples();
  for (let y = y1; y < y2; y += 1) {
    for (let x = x1; x < x2; x += 1) {
      const maskIndex = y * imageWidth + x;
      if (ringMask[maskIndex] === 0 || groupMask[maskIndex] > 0 || pageTextMask[maskIndex] > 0) {
        continue;
      }
      pushSample(samples, imageData, maskIndex);
    }
  }
  return samples;
}

function collectRegionSamples({
  imageData,
  imageWidth,
  imageHeight,
  pageTextMask,
  regionBox,
}) {
  const [x1, y1, x2, y2] = integerBox(regionBox, imageWidth, imageHeight);
  const samples = emptySamples();
  for (let y = y1; y < y2; y += 1) {
    for (let x = x1; x < x2; x += 1) {
      const maskIndex = y * imageWidth + x;
      if (pageTextMask[maskIndex] > 0) {
        continue;
      }
      pushSample(samples, imageData, maskIndex);
    }
  }
  return samples;
}

function dilateMaskBuffer({ maskBuffer, width, height, padding }) {
  if (padding <= 0) {
    return Buffer.from(maskBuffer);
  }
  const mat = new cv.Mat(height, width, cv.CV_8UC1);
  const kernelSize = Math.max(1, Math.round(padding) * 2 + 1);
  const kernel = cv.Mat.ones(kernelSize, kernelSize, cv.CV_8UC1);
  const dilated = new cv.Mat();
  try {
    mat.data.set(maskBuffer);
    cv.dilate(mat, dilated, kernel);
    return Buffer.from(dilated.data);
  } finally {
    mat.delete();
    kernel.delete();
    dilated.delete();
  }
}

function emptySamples() {
  return {
    r: [],
    g: [],
    b: [],
    count: 0,
  };
}

function pushSample(samples, imageData, maskIndex) {
  const imageIndex = maskIndex * 3;
  samples.r.push(imageData[imageIndex]);
  samples.g.push(imageData[imageIndex + 1]);
  samples.b.push(imageData[imageIndex + 2]);
  samples.count += 1;
}

function medianSampleColor(samples) {
  return [
    median(samples.r),
    median(samples.g),
    median(samples.b),
  ];
}

function median(values) {
  if (values.length === 0) {
    return 255;
  }
  values.sort((a, b) => a - b);
  return values[Math.floor(values.length / 2)];
}

function fillGroupInImage({ outputData, imageWidth, groupMask, color }) {
  const pixelCount = groupMask.length;
  for (let maskIndex = 0; maskIndex < pixelCount; maskIndex += 1) {
    if (groupMask[maskIndex] === 0) {
      continue;
    }
    const imageIndex = maskIndex * 3;
    outputData[imageIndex] = color[0];
    outputData[imageIndex + 1] = color[1];
    outputData[imageIndex + 2] = color[2];
  }
}

async function drawDebugImage({
  imagePath,
  outputPath,
  imageWidth,
  imageHeight,
  pageMask,
  comicDetections,
  mediumDetBoxes,
  fillGroups,
}) {
  const maskOverlay = Buffer.alloc(imageWidth * imageHeight * 4, 0);
  for (let index = 0; index < pageMask.length; index += 1) {
    if (pageMask[index] === 0) {
      continue;
    }
    const target = index * 4;
    maskOverlay[target] = 220;
    maskOverlay[target + 1] = 38;
    maskOverlay[target + 2] = 38;
    maskOverlay[target + 3] = 72;
  }

  const overlays = [];
  for (const detection of comicDetections) {
    overlays.push(rectSvg({
      box: detection.bbox,
      color: labelColor(detection.label),
      strokeWidth: 2,
      fillOpacity: 0,
    }));
  }
  for (const box of mediumDetBoxes) {
    overlays.push(polygonSvg({
      points: box.pointsInImage,
      color: "#D97706",
      strokeWidth: 1.5,
      fillOpacity: 0,
    }));
  }
  for (const group of fillGroups) {
    if (group.skipped || !group.maskBox) {
      continue;
    }
    overlays.push(rectSvg({
      box: expandBox(group.maskBox, group.ringPaddingUsed || RING_PADDING),
      color: "#9333EA",
      strokeWidth: 1.5,
      dash: "6 4",
      fillOpacity: 0,
    }));
    overlays.push(rectSvg({
      box: group.maskBox,
      color: "#DC2626",
      strokeWidth: 2,
      fillOpacity: 0,
    }));
    overlays.push(labelSvg({
      x: group.maskBox[0],
      y: group.maskBox[1],
      text: `#${group.index + 1} ${group.region.label} rgb(${group.fillColor?.join(",") ?? "?"})`,
      color: "#DC2626",
    }));
  }

  const svg = `<svg width="${imageWidth}" height="${imageHeight}" viewBox="0 0 ${imageWidth} ${imageHeight}" xmlns="http://www.w3.org/2000/svg">${overlays.join("")}</svg>`;
  await sharp(imagePath)
    .composite([
      {
        input: maskOverlay,
        raw: {
          width: imageWidth,
          height: imageHeight,
          channels: 4,
        },
      },
      { input: Buffer.from(svg), top: 0, left: 0 },
    ])
    .toFile(outputPath);
}

function rectSvg({ box, color, strokeWidth, fillOpacity, dash }) {
  const [x1, y1, x2, y2] = box;
  const dashAttribute = dash ? ` stroke-dasharray="${dash}"` : "";
  const fill = fillOpacity ? ` fill="${color}" fill-opacity="${fillOpacity}"` : ' fill="none"';
  return `<rect x="${x1}" y="${y1}" width="${Math.max(0, x2 - x1)}" height="${Math.max(0, y2 - y1)}"${fill} stroke="${color}" stroke-width="${strokeWidth}"${dashAttribute}/>`;
}

function polygonSvg({ points, color, strokeWidth, fillOpacity }) {
  const pointText = points.map(([x, y]) => `${x},${y}`).join(" ");
  const fill = fillOpacity ? ` fill="${color}" fill-opacity="${fillOpacity}"` : ' fill="none"';
  return `<polygon points="${pointText}"${fill} stroke="${color}" stroke-width="${strokeWidth}"/>`;
}

function labelSvg({ x, y, text, color }) {
  const safeText = escapeXml(text);
  const labelY = Math.max(18, y - 4);
  const width = Math.min(640, Math.max(40, countCodePoints(text) * 8 + 8));
  return `<rect x="${x}" y="${labelY - 18}" width="${width}" height="18" fill="${color}" fill-opacity="0.9"/>
<text x="${x + 4}" y="${labelY - 5}" fill="#ffffff" font-size="12" font-family="sans-serif">${safeText}</text>`;
}

function labelColor(label) {
  if (label === "bubble") {
    return "#16A34A";
  }
  if (label === "text_bubble") {
    return "#2563EB";
  }
  return "#D97706";
}

function serializeMediumDetBox(box) {
  return {
    index: box.index,
    score: round(box.score, 6),
    pointsInImage: box.pointsInImage,
    bboxInImage: box.bboxInImage,
    center: box.center,
  };
}

function serializeFillGroup(group) {
  return {
    index: group.index,
    region: group.region,
    boxIndexes: group.boxIndexes,
    boxes: group.boxes,
    maskSourceBox: group.maskSourceBox,
    maskBox: group.maskBox,
    fillColor: group.fillColor,
    fillSource: group.fillSource,
    sampleCount: group.sampleCount,
    ringPaddingUsed: group.ringPaddingUsed,
    skipped: Boolean(group.skipped),
    skipReason: group.skipReason,
  };
}

async function readJson(jsonPath) {
  return JSON.parse(await fs.readFile(jsonPath, "utf8"));
}

async function readDetectionConfig(ymlPath) {
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

async function assertFile(filePath, hint) {
  try {
    await fs.access(filePath);
  } catch {
    throw new Error(`Missing required file: ${filePath}${hint ? ` (${hint})` : ""}`);
  }
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

function normalizeComicImageSize(preprocessorConfig) {
  const size = preprocessorConfig.size ?? {};
  return {
    width: Number(size.width ?? 640),
    height: Number(size.height ?? 640),
  };
}

function normalizeId2Label(id2label) {
  const labels = {};
  for (const [id, label] of Object.entries(id2label ?? {})) {
    labels[Number(id)] = String(label);
  }
  return labels;
}

function pointsToBbox(points, imageWidth, imageHeight, padding = 0) {
  const xs = points.map(([x]) => x);
  const ys = points.map(([, y]) => y);
  return integerBox([
    Math.min(...xs) - padding,
    Math.min(...ys) - padding,
    Math.max(...xs) + padding,
    Math.max(...ys) + padding,
  ], imageWidth, imageHeight);
}

function boxToPoints(box) {
  return [
    [box[0], box[1]],
    [box[2], box[1]],
    [box[2], box[3]],
    [box[0], box[3]],
  ];
}

function maskBufferBbox(mask, width, height) {
  let x1 = width;
  let y1 = height;
  let x2 = 0;
  let y2 = 0;
  for (let y = 0; y < height; y += 1) {
    const row = y * width;
    for (let x = 0; x < width; x += 1) {
      if (mask[row + x] === 0) {
        continue;
      }
      x1 = Math.min(x1, x);
      y1 = Math.min(y1, y);
      x2 = Math.max(x2, x + 1);
      y2 = Math.max(y2, y + 1);
    }
  }
  if (x2 <= x1 || y2 <= y1) {
    return null;
  }
  return [x1, y1, x2, y2];
}

function integerBox(box, imageWidth, imageHeight) {
  const x1 = clamp(Math.floor(box[0]), 0, imageWidth - 1);
  const y1 = clamp(Math.floor(box[1]), 0, imageHeight - 1);
  const x2 = clamp(Math.ceil(box[2]), x1 + 1, imageWidth);
  const y2 = clamp(Math.ceil(box[3]), y1 + 1, imageHeight);
  return [x1, y1, x2, y2];
}

function expandBox(box, padding) {
  return [
    box[0] - padding,
    box[1] - padding,
    box[2] + padding,
    box[3] + padding,
  ];
}

function unionBoxes(a, b) {
  return [
    Math.min(a[0], b[0]),
    Math.min(a[1], b[1]),
    Math.max(a[2], b[2]),
    Math.max(a[3], b[3]),
  ];
}

function bboxArea(box) {
  return Math.max(0, box[2] - box[0]) * Math.max(0, box[3] - box[1]);
}

function bboxCenter(box) {
  return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2];
}

function bboxDistance(a, b) {
  const dx = Math.max(0, Math.max(a[0] - b[2], b[0] - a[2]));
  const dy = Math.max(0, Math.max(a[1] - b[3], b[1] - a[3]));
  return Math.hypot(dx, dy);
}

function pointInBox(point, box) {
  return point[0] >= box[0] && point[0] <= box[2] && point[1] >= box[1] && point[1] <= box[3];
}

function compareDetectedBoxReadingOrder(a, b) {
  const aY = a.bboxInImage[1];
  const bY = b.bboxInImage[1];
  if (Math.abs(aY - bY) > 20) {
    return aY - bY;
  }
  return a.bboxInImage[0] - b.bboxInImage[0];
}

function compareFillGroupReadingOrder(a, b) {
  const aY = a.maskSourceBox[1];
  const bY = b.maskSourceBox[1];
  if (Math.abs(aY - bY) > 20) {
    return aY - bY;
  }
  return a.maskSourceBox[0] - b.maskSourceBox[0];
}

function countByLabel(detections) {
  const counts = {};
  for (const detection of detections) {
    counts[detection.label] = (counts[detection.label] ?? 0) + 1;
  }
  return counts;
}

function readStringListEnv(name, fallback) {
  const value = process.env[name];
  if (value == null || value.trim() === "") {
    return fallback;
  }
  return value
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean);
}

function readNumberEnv(name, fallback) {
  const value = process.env[name];
  if (value == null || value.trim() === "") {
    return fallback;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function countCodePoints(value) {
  return Array.from(String(value)).length;
}

function escapeXml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function round(value, digits = 6) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}
