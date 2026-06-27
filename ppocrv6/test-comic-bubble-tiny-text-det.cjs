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

const TINY_DET_MODEL_DIR = path.join(SCRIPT_DIR, "model", "tiny-det");
const TINY_DET_MODEL_ONNX_PATH = path.join(TINY_DET_MODEL_DIR, "inference.onnx");
const TINY_DET_MODEL_YML_PATH = path.join(TINY_DET_MODEL_DIR, "inference.yml");

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
const COMIC_LABEL_FILTER = (process.env.COMIC_BUBBLE_LABELS ?? "text_bubble,text_free")
  .split(",")
  .map((label) => label.trim())
  .filter(Boolean);
const CANDIDATE_CROP_PADDING = readNumberEnv("COMIC_TEXT_PADDING", 8);

const TINY_DET_THRESH = readNumberEnv("TINY_DET_THRESH", 0.05);
const TINY_DET_BOX_THRESH = readNumberEnv("TINY_DET_BOX_THRESH", 0.1);
const TINY_DET_MAX_CANDIDATES = readNumberEnv("TINY_DET_MAX_CANDIDATES", 3000);
const TINY_DET_UNCLIP_RATIO = readNumberEnv("TINY_DET_UNCLIP_RATIO", null);
const TINY_DET_MIN_SIZE = readNumberEnv("TINY_DET_MIN_SIZE", 3);
const TEXT_BOX_PADDING = readNumberEnv("OCR_TEXT_BOX_PADDING", 2);
const VERTICAL_ASPECT_RATIO = readNumberEnv("OCR_VERTICAL_ASPECT_RATIO", 1.35);

const CLIPPER_SCALE = 1024;

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
});

async function main() {
  const inputImagePaths = INPUT_IMAGE_PATHS.length > 0 ? INPUT_IMAGE_PATHS : DEFAULT_INPUT_IMAGE_PATHS;

  console.time("opencv");
  await waitForOpenCv();
  console.timeEnd("opencv");

  const [comicModelConfig, comicPreprocessorConfig, rawTinyDetConfig] = await Promise.all([
    readJson(COMIC_MODEL_CONFIG_PATH),
    readJson(COMIC_PREPROCESSOR_CONFIG_PATH),
    readDetectionConfig(TINY_DET_MODEL_YML_PATH),
    assertFile(COMIC_MODEL_ONNX_PATH, "make comic-bubble"),
    assertFile(TINY_DET_MODEL_ONNX_PATH, "make tiny-det"),
  ]);
  const tinyDetConfig = withLowScorePostprocess(rawTinyDetConfig);

  const comicImageSize = normalizeComicImageSize(comicPreprocessorConfig);
  const comicId2Label = normalizeId2Label(comicModelConfig.id2label);

  console.time("comic-session");
  const comicSession = await createSession(COMIC_MODEL_ONNX_PATH);
  console.timeEnd("comic-session");

  console.time("tiny-det-session");
  const tinyDetSession = await createSession(TINY_DET_MODEL_ONNX_PATH);
  console.timeEnd("tiny-det-session");

  const comicInputName = comicSession.inputNames.includes("images")
    ? "images"
    : comicSession.inputNames[0];
  const comicTargetSizeInputName = comicSession.inputNames.includes("orig_target_sizes")
    ? "orig_target_sizes"
    : comicSession.inputNames[1];
  const tinyDetInputName = tinyDetSession.inputNames[0] ?? "x";
  const tinyDetOutputName = tinyDetSession.outputNames[0] ?? "fetch_name_0";

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
      tinyDetSession,
      tinyDetInputName,
      tinyDetOutputName,
      tinyDetConfig,
    });
  }
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
  tinyDetSession,
  tinyDetInputName,
  tinyDetOutputName,
  tinyDetConfig,
}) {
  const imageBaseName = path.basename(imagePath, path.extname(imagePath));
  const outputJsonPath = path.join(ASSETS_DIR, `${imageBaseName}.comic-bubble-tiny-text-det.json`);
  const debugImagePath = path.join(ASSETS_DIR, `${imageBaseName}.comic-bubble-tiny-text-det.png`);
  const cropDir = path.join(ASSETS_DIR, `${imageBaseName}.comic-bubble-tiny-text-det-crops`);

  console.log(`\n${path.relative(SCRIPT_DIR, imagePath)}`);

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

  const allDetections = decodeComicDetections(
    comicOutputs,
    comicId2Label,
    comicImage.originalWidth,
    comicImage.originalHeight,
  )
    .filter((detection) => detection.score >= COMIC_SCORE_THRESHOLD)
    .map((detection, index) => ({
      index,
      ...detection,
      bbox: detection.bbox.map((value) => round(value, 3)),
    }));

  const labelFilter = new Set(COMIC_LABEL_FILTER);
  const candidateDetections = allDetections
    .filter((detection) => labelFilter.size === 0 || labelFilter.has(detection.label))
    .sort(compareDetectionReadingOrder);

  await fs.mkdir(cropDir, { recursive: true });

  console.time(`${imageBaseName}:tiny-det`);
  const textAreas = [];
  for (let index = 0; index < candidateDetections.length; index += 1) {
    const detection = candidateDetections[index];
    const textArea = await detectTextArea({
      imagePath,
      imageBaseName,
      cropDir,
      areaIndex: index,
      detection,
      tinyDetSession,
      tinyDetInputName,
      tinyDetOutputName,
      tinyDetConfig,
    });
    textAreas.push(textArea);
  }
  console.timeEnd(`${imageBaseName}:tiny-det`);

  const result = {
    image: path.resolve(imagePath),
    models: {
      comicBubble: {
        repository: "ogkalu/comic-text-and-bubble-detector",
        variant: "detector-v4-s_int8.onnx",
        onnx: COMIC_MODEL_ONNX_PATH,
        config: COMIC_MODEL_CONFIG_PATH,
        preprocessor: COMIC_PREPROCESSOR_CONFIG_PATH,
      },
      textDetection: {
        repository: "PaddlePaddle/PP-OCRv6_tiny_det_onnx",
        onnx: TINY_DET_MODEL_ONNX_PATH,
        yml: TINY_DET_MODEL_YML_PATH,
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
        inputNames: tinyDetSession.inputNames,
        outputNames: tinyDetSession.outputNames,
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
        imgMode: tinyDetConfig.preprocessing.imgMode,
        scale: tinyDetConfig.preprocessing.scale,
        mean: tinyDetConfig.preprocessing.mean,
        std: tinyDetConfig.preprocessing.std,
        detResizeForTest: tinyDetConfig.preprocessing.detResizeForTest,
      },
    },
    postprocess: {
      comicBubble: {
        threshold: COMIC_SCORE_THRESHOLD,
        labelFilter: COMIC_LABEL_FILTER,
        boxFormat: "xyxy",
      },
      textDetection: {
        ...tinyDetConfig.postprocess,
        mode: "opencv-dbpostprocess-quad",
        candidateCropPadding: CANDIDATE_CROP_PADDING,
        textBoxPadding: TEXT_BOX_PADDING,
        verticalAspectRatio: VERTICAL_ASPECT_RATIO,
      },
    },
    counts: {
      comicDetections: countByLabel(allDetections),
      candidateAreas: textAreas.length,
      textBoxes: textAreas.reduce((sum, area) => sum + area.textDetection.boxCount, 0),
    },
    detections: allDetections,
    textAreas,
  };

  await fs.writeFile(outputJsonPath, `${JSON.stringify(result, null, 2)}\n`, "utf8");
  await drawDebugImage(imagePath, debugImagePath, textAreas);

  console.log(`comic detections: ${allDetections.length}`, result.counts.comicDetections);
  console.log(`candidate areas: ${textAreas.length}`);
  for (const textArea of textAreas) {
    console.log(
      `  #${textArea.index + 1} ${textArea.detection.label} score=${textArea.detection.score.toFixed(3)} boxes=${textArea.textDetection.boxCount}`,
    );
  }
  console.log(`text boxes: ${result.counts.textBoxes}`);
  console.log(`json: ${path.relative(SCRIPT_DIR, outputJsonPath)}`);
  console.log(`debug: ${path.relative(SCRIPT_DIR, debugImagePath)}`);
  console.log(`crops: ${path.relative(SCRIPT_DIR, cropDir)}`);
}

async function detectTextArea({
  imagePath,
  imageBaseName,
  cropDir,
  areaIndex,
  detection,
  tinyDetSession,
  tinyDetInputName,
  tinyDetOutputName,
  tinyDetConfig,
}) {
  const crop = await cropImage(imagePath, detection.bbox, CANDIDATE_CROP_PADDING);
  const areaCropPath = path.join(cropDir, `${imageBaseName}.area-${areaIndex + 1}.candidate.png`);
  await fs.writeFile(areaCropPath, crop.buffer);

  const textBoxes = await detectTextBoxesInCrop({
    crop,
    tinyDetSession,
    tinyDetInputName,
    tinyDetOutputName,
    tinyDetConfig,
  });

  return {
    index: areaIndex,
    sourceDetectionIndex: detection.index,
    detection,
    crop: {
      path: areaCropPath,
      padding: CANDIDATE_CROP_PADDING,
      bboxInImage: crop.bboxInImage.map((value) => round(value, 3)),
      width: crop.width,
      height: crop.height,
    },
    textDetection: {
      boxCount: textBoxes.length,
      boxes: textBoxes.map(serializeDetectedBox),
    },
  };
}

async function detectTextBoxesInCrop({
  crop,
  tinyDetSession,
  tinyDetInputName,
  tinyDetOutputName,
  tinyDetConfig,
}) {
  const preprocessed = await preprocessDetectionImage(crop.buffer, tinyDetConfig.preprocessing);
  const inputTensor = new ort.Tensor("float32", preprocessed.tensorData, [
    1,
    3,
    preprocessed.height,
    preprocessed.width,
  ]);
  const outputs = await tinyDetSession.run({ [tinyDetInputName]: inputTensor });
  const outputTensor = outputs[tinyDetOutputName] ?? outputs[tinyDetSession.outputNames[0]];
  if (!outputTensor) {
    throw new Error(`Detection model output not found. Available outputs: ${Object.keys(outputs).join(", ")}`);
  }

  return detectBoxes(
    outputTensor,
    preprocessed.originalWidth,
    preprocessed.originalHeight,
    tinyDetConfig.postprocess,
  )
    .map((box, index) => {
      const bbox = pointsToBbox(box.points, crop.width, crop.height, TEXT_BOX_PADDING);
      return {
        index,
        score: box.score,
        pointsInCrop: box.points.map((point) => point.map((value) => round(value, 3))),
        pointsInImage: box.points.map(([x, y]) => [
          round(x + crop.bboxInImage[0], 3),
          round(y + crop.bboxInImage[1], 3),
        ]),
        bboxInCrop: bbox,
        bboxInImage: offsetBox(bbox, crop.bboxInImage[0], crop.bboxInImage[1]),
      };
    })
    .filter((box) => isUsableDetectedBox(box.bboxInCrop))
    .sort(compareDetectedBoxReadingOrder);
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

async function preprocessDetectionImage(imageBuffer, preprocessing) {
  const metadata = await sharp(imageBuffer).metadata();
  if (!metadata.width || !metadata.height) {
    throw new Error("Unable to read detection crop dimensions.");
  }

  const size = resizeForTest(metadata.width, metadata.height, preprocessing.detResizeForTest);
  const { data } = await sharp(imageBuffer)
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

async function cropImage(imagePath, bbox, padding) {
  const metadata = await sharp(imagePath).metadata();
  if (!metadata.width || !metadata.height) {
    throw new Error(`Unable to read image dimensions: ${imagePath}`);
  }

  const [x1, y1, x2, y2] = bbox;
  const left = clamp(Math.floor(x1 - padding), 0, metadata.width - 1);
  const top = clamp(Math.floor(y1 - padding), 0, metadata.height - 1);
  const right = clamp(Math.ceil(x2 + padding), left + 1, metadata.width);
  const bottom = clamp(Math.ceil(y2 + padding), top + 1, metadata.height);
  const width = right - left;
  const height = bottom - top;
  const buffer = await sharp(imagePath)
    .extract({ left, top, width, height })
    .png()
    .toBuffer();

  return {
    buffer,
    width,
    height,
    bboxInImage: [left, top, right, bottom],
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

async function drawDebugImage(imagePath, outputPath, textAreas) {
  const metadata = await sharp(imagePath).metadata();
  if (!metadata.width || !metadata.height) {
    throw new Error(`Unable to read image dimensions: ${imagePath}`);
  }

  const overlays = [];
  for (const textArea of textAreas) {
    const [x1, y1, x2, y2] = textArea.detection.bbox;
    const [cx1, cy1, cx2, cy2] = textArea.crop.bboxInImage;
    const label = `#${textArea.index + 1} ${textArea.detection.label} ${textArea.detection.score.toFixed(2)} boxes=${textArea.textDetection.boxCount}`;

    overlays.push(rectSvg({
      x1: cx1,
      y1: cy1,
      x2: cx2,
      y2: cy2,
      color: "#2563EB",
      strokeWidth: 2,
      dash: "6 4",
    }));
    overlays.push(rectSvg({
      x1,
      y1,
      x2,
      y2,
      color: "#2563EB",
      strokeWidth: 3,
    }));
    overlays.push(labelSvg({
      x: x1,
      y: y1,
      text: label,
      color: "#2563EB",
    }));

    for (const box of textArea.textDetection.boxes) {
      const color = box.orientation === "vertical" ? "#DC2626" : "#D97706";
      overlays.push(polygonSvg({
        points: box.pointsInImage,
        color,
        strokeWidth: 2,
        fillOpacity: 0.14,
      }));
      overlays.push(labelSvg({
        x: box.bboxInImage[0],
        y: box.bboxInImage[1],
        text: `${box.index + 1}:${box.score.toFixed(2)}`,
        color,
        compact: true,
      }));
    }
  }

  const svg = `<svg width="${metadata.width}" height="${metadata.height}" viewBox="0 0 ${metadata.width} ${metadata.height}" xmlns="http://www.w3.org/2000/svg">${overlays.join("")}</svg>`;
  await sharp(imagePath).composite([{ input: Buffer.from(svg), top: 0, left: 0 }]).toFile(outputPath);
}

function rectSvg({ x1, y1, x2, y2, color, strokeWidth, dash }) {
  const dashAttribute = dash ? ` stroke-dasharray="${dash}"` : "";
  return `<rect x="${x1}" y="${y1}" width="${Math.max(0, x2 - x1)}" height="${Math.max(0, y2 - y1)}" fill="none" stroke="${color}" stroke-width="${strokeWidth}"${dashAttribute}/>`;
}

function polygonSvg({ points, color, strokeWidth, fillOpacity }) {
  const pointText = points.map(([x, y]) => `${x},${y}`).join(" ");
  const fill = fillOpacity ? ` fill="${color}" fill-opacity="${fillOpacity}"` : ' fill="none"';
  return `<polygon points="${pointText}"${fill} stroke="${color}" stroke-width="${strokeWidth}"/>`;
}

function labelSvg({ x, y, text, color, compact = false }) {
  const safeText = escapeXml(text);
  const labelY = Math.max(compact ? 12 : 18, y - 4);
  const fontSize = compact ? 10 : 12;
  const height = compact ? 16 : 20;
  const width = Math.min(640, Math.max(compact ? 28 : 40, countCodePoints(text) * (compact ? 6 : 8) + 8));
  return `<rect x="${x}" y="${labelY - height + 4}" width="${width}" height="${height}" fill="${color}" fill-opacity="0.9"/>
<text x="${x + 4}" y="${labelY - 4}" fill="#ffffff" font-size="${fontSize}" font-family="sans-serif">${safeText}</text>`;
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
      minSize: TINY_DET_MIN_SIZE,
    },
  };
}

function withLowScorePostprocess(config) {
  return {
    ...config,
    postprocess: {
      ...config.postprocess,
      thresh: TINY_DET_THRESH,
      boxThresh: TINY_DET_BOX_THRESH,
      maxCandidates: TINY_DET_MAX_CANDIDATES,
      unclipRatio: TINY_DET_UNCLIP_RATIO ?? config.postprocess.unclipRatio,
      minSize: TINY_DET_MIN_SIZE,
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

function serializeDetectedBox(box) {
  return {
    index: box.index,
    score: round(box.score, 6),
    pointsInCrop: box.pointsInCrop,
    pointsInImage: box.pointsInImage,
    bboxInCrop: box.bboxInCrop.map((value) => round(value, 3)),
    bboxInImage: box.bboxInImage.map((value) => round(value, 3)),
    orientation: isVerticalBbox(box.bboxInCrop) ? "vertical" : "horizontal",
  };
}

function pointsToBbox(points, imageWidth, imageHeight, padding) {
  const xs = points.map(([x]) => x);
  const ys = points.map(([, y]) => y);
  return [
    clamp(Math.floor(Math.min(...xs) - padding), 0, imageWidth),
    clamp(Math.floor(Math.min(...ys) - padding), 0, imageHeight),
    clamp(Math.ceil(Math.max(...xs) + padding), 0, imageWidth),
    clamp(Math.ceil(Math.max(...ys) + padding), 0, imageHeight),
  ];
}

function isUsableDetectedBox(box) {
  return box[2] - box[0] >= 3 && box[3] - box[1] >= 3;
}

function isVerticalBbox(box) {
  const width = Math.max(1, box[2] - box[0]);
  const height = Math.max(1, box[3] - box[1]);
  return height / width >= VERTICAL_ASPECT_RATIO;
}

function compareDetectedBoxReadingOrder(a, b) {
  const aVertical = isVerticalBbox(a.bboxInCrop);
  const bVertical = isVerticalBbox(b.bboxInCrop);
  if (aVertical !== bVertical) {
    return aVertical ? -1 : 1;
  }
  return aVertical ? compareVerticalBoxes(a, b) : compareHorizontalBoxes(a, b);
}

function compareHorizontalBoxes(a, b) {
  const aBox = a.bboxInCrop;
  const bBox = b.bboxInCrop;
  const aY = aBox[1];
  const bY = bBox[1];
  if (Math.abs(aY - bY) > 12) {
    return aY - bY;
  }
  return aBox[0] - bBox[0];
}

function compareVerticalBoxes(a, b) {
  const aBox = a.bboxInCrop;
  const bBox = b.bboxInCrop;
  const aX = aBox[0];
  const bX = bBox[0];
  if (Math.abs(aX - bX) > 8) {
    return bX - aX;
  }
  return aBox[1] - bBox[1];
}

function compareDetectionReadingOrder(a, b) {
  const aY = a.bbox[1];
  const bY = b.bbox[1];
  if (Math.abs(aY - bY) > 20) {
    return aY - bY;
  }
  return a.bbox[0] - b.bbox[0];
}

function offsetBox(box, offsetX, offsetY) {
  return [
    box[0] + offsetX,
    box[1] + offsetY,
    box[2] + offsetX,
    box[3] + offsetY,
  ];
}

async function readJson(jsonPath) {
  return JSON.parse(await fs.readFile(jsonPath, "utf8"));
}

async function assertFile(filePath, hint) {
  try {
    await fs.access(filePath);
  } catch {
    throw new Error(`Missing required file: ${filePath}${hint ? ` (${hint})` : ""}`);
  }
}

function countByLabel(detections) {
  const counts = {};
  for (const detection of detections) {
    counts[detection.label] = (counts[detection.label] ?? 0) + 1;
  }
  return counts;
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
