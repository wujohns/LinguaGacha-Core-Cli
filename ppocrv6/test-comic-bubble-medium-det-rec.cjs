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

const REC_MODEL_DIR = path.join(SCRIPT_DIR, "model", "medium-rec");
const REC_MODEL_ONNX_PATH = path.join(REC_MODEL_DIR, "inference.onnx");
const REC_MODEL_YML_PATH = path.join(REC_MODEL_DIR, "inference.yml");

const DEFAULT_INPUT_IMAGE_PATHS = [
  // path.join(ASSETS_DIR, "comic_en.png"),
  // path.join(ASSETS_DIR, "comic_jp.png"),
  path.join(ASSETS_DIR, "cct.png"),
  path.join(ASSETS_DIR, "cctt.png"),
];

const INPUT_IMAGE_PATHS = (process.env.INPUT_IMAGES ?? "")
  .split(",")
  .map((entry) => entry.trim())
  .filter(Boolean)
  .map((entry) => path.resolve(SCRIPT_DIR, entry));

const COMIC_SCORE_THRESHOLD = Number(process.env.COMIC_BUBBLE_THRESHOLD ?? "0.35");
const COMIC_TEXT_LABEL = process.env.COMIC_TEXT_LABEL ?? "text_bubble";
const BUBBLE_CROP_PADDING = Number(process.env.COMIC_TEXT_PADDING ?? "8");
const OCR_BOX_PADDING = Number(process.env.OCR_TEXT_BOX_PADDING ?? "2");
const VERTICAL_ASPECT_RATIO = Number(process.env.OCR_VERTICAL_ASPECT_RATIO ?? "1.35");

const MIN_SIZE = 3;
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

  const [comicModelConfig, comicPreprocessorConfig, detConfig, recConfig] = await Promise.all([
    readJson(COMIC_MODEL_CONFIG_PATH),
    readJson(COMIC_PREPROCESSOR_CONFIG_PATH),
    readDetectionConfig(DET_MODEL_YML_PATH),
    readRecognitionConfig(REC_MODEL_YML_PATH),
    assertFile(COMIC_MODEL_ONNX_PATH, "make comic-bubble"),
    assertFile(DET_MODEL_ONNX_PATH, "make medium-det"),
    assertFile(REC_MODEL_ONNX_PATH, "make medium-rec"),
  ]);

  const comicImageSize = normalizeComicImageSize(comicPreprocessorConfig);
  const comicId2Label = normalizeId2Label(comicModelConfig.id2label);

  console.time("comic-session");
  const comicSession = await createSession(COMIC_MODEL_ONNX_PATH);
  console.timeEnd("comic-session");

  console.time("det-session");
  const detSession = await createSession(DET_MODEL_ONNX_PATH);
  console.timeEnd("det-session");

  console.time("rec-session");
  const recSession = await createSession(REC_MODEL_ONNX_PATH);
  console.timeEnd("rec-session");

  const comicInputName = comicSession.inputNames.includes("images")
    ? "images"
    : comicSession.inputNames[0];
  const comicTargetSizeInputName = comicSession.inputNames.includes("orig_target_sizes")
    ? "orig_target_sizes"
    : comicSession.inputNames[1];
  const detInputName = detSession.inputNames[0] ?? "x";
  const detOutputName = detSession.outputNames[0] ?? "fetch_name_0";
  const recInputName = recSession.inputNames[0] ?? "x";
  const recOutputName = recSession.outputNames[0] ?? "fetch_name_0";

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
      recSession,
      recInputName,
      recOutputName,
      recConfig,
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
  detSession,
  detInputName,
  detOutputName,
  detConfig,
  recSession,
  recInputName,
  recOutputName,
  recConfig,
}) {
  const imageBaseName = path.basename(imagePath, path.extname(imagePath));
  const outputJsonPath = path.join(ASSETS_DIR, `${imageBaseName}.comic-bubble-medium-det-rec.json`);
  const debugImagePath = path.join(ASSETS_DIR, `${imageBaseName}.comic-bubble-medium-det-rec.png`);
  const cropDir = path.join(ASSETS_DIR, `${imageBaseName}.comic-bubble-medium-det-rec-crops`);

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

  const textDetections = allDetections
    .filter((detection) => detection.label === COMIC_TEXT_LABEL)
    .sort(compareDetectionReadingOrder);

  await fs.mkdir(cropDir, { recursive: true });

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
    .map((box, index) => normalizeImageDetectedBox(box, index, detImage.originalWidth, detImage.originalHeight))
    .filter((box) => isUsableDetectedBox(box.bboxInImage))
    .sort(compareImageDetectedBoxReadingOrder)
    .map((box, index) => ({
      ...box,
      index,
    }));
  console.timeEnd(`${imageBaseName}:medium-det-postprocess`);

  console.time(`${imageBaseName}:rec`);
  const textAreas = [];
  for (let index = 0; index < textDetections.length; index += 1) {
    const detection = textDetections[index];
    const textArea = await recognizeTextArea({
      imagePath,
      imageBaseName,
      cropDir,
      textAreaIndex: index,
      detection,
      mediumDetBoxes,
      recSession,
      recInputName,
      recOutputName,
      recConfig,
    });
    textAreas.push(textArea);
  }
  console.timeEnd(`${imageBaseName}:rec`);

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
        repository: "PaddlePaddle/PP-OCRv6_medium_det_onnx",
        onnx: DET_MODEL_ONNX_PATH,
        yml: DET_MODEL_YML_PATH,
      },
      recognition: {
        repository: "PaddlePaddle/PP-OCRv6_medium_rec_onnx",
        onnx: REC_MODEL_ONNX_PATH,
        yml: REC_MODEL_YML_PATH,
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
      recognition: {
        inputNames: recSession.inputNames,
        outputNames: recSession.outputNames,
        inputShape: [1, ...recConfig.imageShape],
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
      recognition: {
        imgMode: recConfig.imgMode,
        imageShape: recConfig.imageShape,
        scale: recConfig.scale,
        mean: recConfig.mean,
        std: recConfig.std,
        characterCount: recConfig.characterDict.length,
      },
    },
    postprocess: {
      comicBubble: {
        threshold: COMIC_SCORE_THRESHOLD,
        textLabel: COMIC_TEXT_LABEL,
        boxFormat: "xyxy",
      },
      textDetection: {
        ...detConfig.postprocess,
        mode: "opencv-dbpostprocess-quad",
        detectionScope: "whole-image-once",
        assignmentMode: "center-in-text-area-crop",
        bubbleCropPadding: BUBBLE_CROP_PADDING,
        recognitionBoxPadding: OCR_BOX_PADDING,
        verticalAspectRatio: VERTICAL_ASPECT_RATIO,
      },
      recognition: {
        name: recConfig.postprocessName,
        decoder: "ctc-greedy",
      },
    },
    counts: {
      comicDetections: countByLabel(allDetections),
      textAreas: textAreas.length,
      mediumDetBoxes: mediumDetBoxes.length,
      assignedMediumDetBoxes: countUniqueAssignedBoxes(textAreas),
      recognizedSegments: textAreas.reduce(
        (sum, textArea) => sum + textArea.recognition.selected.segmentCount,
        0,
      ),
    },
    detections: allDetections,
    textDetection: {
      boxCount: mediumDetBoxes.length,
      boxes: mediumDetBoxes.map(serializeImageDetectedBox),
    },
    textAreas,
  };

  await fs.writeFile(outputJsonPath, `${JSON.stringify(result, null, 2)}\n`, "utf8");
  await drawDebugImage(imagePath, debugImagePath, textAreas);

  console.log(`comic detections: ${allDetections.length}`, result.counts.comicDetections);
  console.log(`medium-det boxes: ${mediumDetBoxes.length}`);
  console.log(`text areas: ${textAreas.length}`);
  for (const textArea of textAreas) {
    const selected = textArea.recognition.selected;
    const text = selected.text.replace(/\s+/g, " ").trim();
    console.log(
      `  #${textArea.index + 1} ${selected.orientation} boxes=${selected.segmentCount} confidence=${selected.confidence.toFixed(3)} text=${JSON.stringify(text)}`,
    );
  }
  console.log(`json: ${path.relative(SCRIPT_DIR, outputJsonPath)}`);
  console.log(`debug: ${path.relative(SCRIPT_DIR, debugImagePath)}`);
  console.log(`crops: ${path.relative(SCRIPT_DIR, cropDir)}`);
}

async function recognizeTextArea({
  imagePath,
  imageBaseName,
  cropDir,
  textAreaIndex,
  detection,
  mediumDetBoxes,
  recSession,
  recInputName,
  recOutputName,
  recConfig,
}) {
  const crop = await cropImage(imagePath, detection.bbox, BUBBLE_CROP_PADDING);
  const areaCropPath = path.join(cropDir, `${imageBaseName}.text-${textAreaIndex + 1}.bubble.png`);
  await fs.writeFile(areaCropPath, crop.buffer);

  const detectedBoxes = findTextBoxesInCrop({
    crop,
    mediumDetBoxes,
  });
  const segments = buildRecognitionSegments(detectedBoxes, crop);
  const segmentsForRecognition = segments.length > 0
    ? segments
    : [{
      index: 0,
      source: "fallback-whole-bubble",
      bboxInCrop: [0, 0, crop.width, crop.height],
      bboxInImage: crop.bboxInImage,
      pointsInCrop: [
        [0, 0],
        [crop.width, 0],
        [crop.width, crop.height],
        [0, crop.height],
      ],
      pointsInImage: [
        [crop.bboxInImage[0], crop.bboxInImage[1]],
        [crop.bboxInImage[2], crop.bboxInImage[1]],
        [crop.bboxInImage[2], crop.bboxInImage[3]],
        [crop.bboxInImage[0], crop.bboxInImage[3]],
      ],
      rotation: 0,
      orientation: "unknown",
      detScore: 0,
    }];

  const selected = await recognizeDetectedSegments({
    sourceBuffer: crop.buffer,
    segments: segmentsForRecognition,
    recSession,
    recInputName,
    recOutputName,
    recConfig,
  });

  await writeSelectedSegmentCrops({
    selected,
    cropDir,
    imageBaseName,
    textAreaIndex,
  });

  return {
    index: textAreaIndex,
    sourceDetectionIndex: detection.index,
    detection,
    crop: {
      path: areaCropPath,
      padding: BUBBLE_CROP_PADDING,
      bboxInImage: crop.bboxInImage.map((value) => round(value, 3)),
      width: crop.width,
      height: crop.height,
    },
    textDetection: {
      boxCount: detectedBoxes.length,
      boxes: detectedBoxes.map(serializeDetectedBox),
      fallbackUsed: detectedBoxes.length === 0,
    },
    recognition: {
      selected: serializeCandidate(selected),
    },
  };
}

function findTextBoxesInCrop({ crop, mediumDetBoxes }) {
  const [cropX1, cropY1, cropX2, cropY2] = crop.bboxInImage;
  return mediumDetBoxes
    .filter((box) => pointInBox(box.center, crop.bboxInImage))
    .map((box) => {
      const pointsInCrop = box.pointsInImage.map(([x, y]) => [
        round(x - cropX1, 3),
        round(y - cropY1, 3),
      ]);
      const bboxInCrop = pointsToBbox(pointsInCrop, crop.width, crop.height, OCR_BOX_PADDING);
      return {
        index: box.index,
        score: box.score,
        pointsInCrop,
        pointsInImage: box.pointsInImage,
        bboxInCrop,
        bboxInImage: [
          clamp(bboxInCrop[0] + cropX1, cropX1, cropX2),
          clamp(bboxInCrop[1] + cropY1, cropY1, cropY2),
          clamp(bboxInCrop[2] + cropX1, cropX1, cropX2),
          clamp(bboxInCrop[3] + cropY1, cropY1, cropY2),
        ],
      };
    })
    .filter((box) => isUsableDetectedBox(box.bboxInCrop))
    .sort(compareDetectedBoxReadingOrder);
}

function buildRecognitionSegments(detectedBoxes, crop) {
  const verticalCount = detectedBoxes.filter((box) => isVerticalBbox(box.bboxInCrop)).length;
  const horizontalCount = detectedBoxes.length - verticalCount;
  const areaOrientation = verticalCount > 0 && verticalCount >= horizontalCount ? "vertical" : "horizontal";
  const sorted = [...detectedBoxes].sort(
    areaOrientation === "vertical" ? compareVerticalBoxes : compareHorizontalBoxes,
  );

  return sorted.map((box, index) => {
    const orientation = isVerticalBbox(box.bboxInCrop) ? "vertical" : "horizontal";
    return {
      index,
      source: "ppocrv6-medium-det",
      detBoxIndex: box.index,
      detScore: box.score,
      orientation,
      rotation: orientation === "vertical" ? 270 : 0,
      bboxInCrop: box.bboxInCrop,
      bboxInImage: box.bboxInImage,
      pointsInCrop: box.pointsInCrop,
      pointsInImage: box.pointsInImage,
      cropSize: {
        width: crop.width,
        height: crop.height,
      },
    };
  });
}

async function recognizeDetectedSegments({
  sourceBuffer,
  segments,
  recSession,
  recInputName,
  recOutputName,
  recConfig,
}) {
  const recognizedSegments = [];
  for (let index = 0; index < segments.length; index += 1) {
    const segment = segments[index];
    const recognized = await recognizeSegment({
      sourceBuffer,
      segment: { ...segment, index },
      recSession,
      recInputName,
      recOutputName,
      recConfig,
    });
    recognizedSegments.push(recognized);
  }

  const orientation = inferCandidateOrientation(recognizedSegments);
  const classifiedSegments = classifyRubySegments(recognizedSegments, orientation);
  const textSegments = classifiedSegments.filter((segment) => !segment.ignored && segment.text.length > 0);
  const joiner = orientation === "vertical" ? "" : "\n";
  const text = textSegments.map((segment) => segment.text).join(joiner);
  const confidence = weightedAverage(
    textSegments.map((segment) => ({
      value: segment.confidence,
      weight: Math.max(1, countCodePoints(segment.text)),
    })),
  );

  return {
    name: "medium-det-boxes",
    orientation,
    text,
    confidence,
    charCount: countCodePoints(text),
    segmentCount: classifiedSegments.length,
    textSegmentCount: textSegments.length,
    ignoredSegmentCount: classifiedSegments.filter((segment) => segment.ignored).length,
    segments: classifiedSegments,
  };
}

async function recognizeSegment({
  sourceBuffer,
  segment,
  recSession,
  recInputName,
  recOutputName,
  recConfig,
}) {
  const segmentBuffer = await extractSegmentBuffer(sourceBuffer, segment.bboxInCrop, segment.rotation);
  const preprocessed = await preprocessRecognitionImage(segmentBuffer, recConfig);
  const inputTensor = new ort.Tensor("float32", preprocessed.tensorData, [
    1,
    recConfig.imageShape[0],
    recConfig.imageShape[1],
    recConfig.imageShape[2],
  ]);
  const outputs = await recSession.run({ [recInputName]: inputTensor });
  const outputTensor = outputs[recOutputName] ?? outputs[recSession.outputNames[0]];
  if (!outputTensor) {
    throw new Error(`Recognition model output not found. Available outputs: ${Object.keys(outputs).join(", ")}`);
  }

  const decoded = decodeCtcGreedy(outputTensor, recConfig.characterDict);

  return {
    index: segment.index,
    source: segment.source,
    detBoxIndex: segment.detBoxIndex ?? null,
    detScore: round(segment.detScore ?? 0, 6),
    orientation: segment.orientation,
    rotation: segment.rotation,
    bboxInCrop: segment.bboxInCrop.map((value) => round(value, 3)),
    bboxInImage: segment.bboxInImage.map((value) => round(value, 3)),
    pointsInCrop: segment.pointsInCrop,
    pointsInImage: segment.pointsInImage,
    text: decoded.text,
    confidence: decoded.confidence,
    tokens: decoded.tokens,
    cropPath: null,
    input: {
      width: preprocessed.width,
      height: preprocessed.height,
      resizedWidth: preprocessed.resizedWidth,
      validRatio: preprocessed.validRatio,
    },
    outputShape: outputTensor.dims,
    imageBuffer: segmentBuffer,
  };
}

async function writeSelectedSegmentCrops({ selected, cropDir, imageBaseName, textAreaIndex }) {
  for (const segment of selected.segments) {
    const segmentPath = path.join(
      cropDir,
      `${imageBaseName}.text-${textAreaIndex + 1}.det-rec-${segment.index + 1}.png`,
    );
    await fs.writeFile(segmentPath, segment.imageBuffer);
    segment.cropPath = segmentPath;
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

async function preprocessRecognitionImage(imageBuffer, recConfig) {
  const metadata = await sharp(imageBuffer).metadata();
  if (!metadata.width || !metadata.height) {
    throw new Error("Unable to read recognition crop dimensions.");
  }

  const [channels, targetHeight, targetWidth] = recConfig.imageShape;
  const resizedWidth = Math.min(
    targetWidth,
    Math.max(1, Math.ceil((metadata.width / metadata.height) * targetHeight)),
  );
  const { data } = await sharp(imageBuffer)
    .resize(resizedWidth, targetHeight, { fit: "fill" })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const planeSize = targetWidth * targetHeight;
  const tensorData = new Float32Array(channels * planeSize);
  for (let y = 0; y < targetHeight; y += 1) {
    for (let x = 0; x < resizedWidth; x += 1) {
      const sourceIndex = (y * resizedWidth + x) * 3;
      const targetIndex = y * targetWidth + x;
      const r = data[sourceIndex];
      const g = data[sourceIndex + 1];
      const b = data[sourceIndex + 2];
      const values = recConfig.imgMode === "BGR" ? [b, g, r] : [r, g, b];
      for (let channel = 0; channel < channels; channel += 1) {
        const scaled = values[channel] * recConfig.scale;
        tensorData[channel * planeSize + targetIndex] =
          (scaled - recConfig.mean[channel]) / recConfig.std[channel];
      }
    }
  }

  return {
    tensorData,
    width: targetWidth,
    height: targetHeight,
    resizedWidth,
    validRatio: resizedWidth / targetWidth,
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

function decodeCtcGreedy(outputTensor, characterDict) {
  const dims = outputTensor.dims;
  if (dims.length !== 3 || dims[0] !== 1) {
    throw new Error(`Unexpected recognition output shape: ${dims.join("x")}`);
  }

  const [, steps, classCount] = dims;
  const characters = buildCtcCharacters(characterDict, classCount);
  let previousClassIndex = -1;
  let text = "";
  const keptScores = [];
  const tokens = [];

  for (let step = 0; step < steps; step += 1) {
    let bestClassIndex = 0;
    let bestScore = -Infinity;
    for (let classIndex = 0; classIndex < classCount; classIndex += 1) {
      const score = Number(outputTensor.data[step * classCount + classIndex]);
      if (score > bestScore) {
        bestScore = score;
        bestClassIndex = classIndex;
      }
    }

    if (bestClassIndex !== 0 && bestClassIndex !== previousClassIndex) {
      const char = characters[bestClassIndex] ?? "";
      if (char.length > 0) {
        text += char;
        keptScores.push(bestScore);
        tokens.push({
          step,
          classIndex: bestClassIndex,
          text: char,
          confidence: round(bestScore, 6),
        });
      }
    }
    previousClassIndex = bestClassIndex;
  }

  return {
    text,
    confidence: keptScores.length > 0
      ? round(keptScores.reduce((sum, score) => sum + score, 0) / keptScores.length, 6)
      : 0,
    tokens,
  };
}

function buildCtcCharacters(characterDict, classCount) {
  if (classCount === characterDict.length + 2) {
    return ["", ...characterDict, " "];
  }
  if (classCount === characterDict.length + 1) {
    return ["", ...characterDict];
  }
  if (classCount === characterDict.length) {
    return characterDict;
  }

  const characters = ["", ...characterDict];
  while (characters.length < classCount) {
    characters.push("");
  }
  return characters.slice(0, classCount);
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

async function extractSegmentBuffer(sourceBuffer, bbox, rotation) {
  const [x1, y1, x2, y2] = bbox.map((value) => Math.round(value));
  const metadata = await sharp(sourceBuffer).metadata();
  if (!metadata.width || !metadata.height) {
    throw new Error("Unable to read segment source dimensions.");
  }

  const left = clamp(x1, 0, metadata.width - 1);
  const top = clamp(y1, 0, metadata.height - 1);
  const right = clamp(x2, left + 1, metadata.width);
  const bottom = clamp(y2, top + 1, metadata.height);
  let image = sharp(sourceBuffer).extract({ left, top, width: right - left, height: bottom - top });
  if (rotation) {
    image = image.rotate(rotation);
  }
  return image.png().toBuffer();
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
    const selected = textArea.recognition.selected;
    const [x1, y1, x2, y2] = textArea.crop.bboxInImage;
    const label = `${textArea.index + 1} ${selected.orientation} ${selected.confidence.toFixed(2)} ${selected.text}`
      .replace(/\s+/g, " ")
      .trim();

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
      overlays.push(polygonSvg({
        points: box.pointsInImage,
        color: "#D97706",
        strokeWidth: 2,
      }));
    }

    const segmentColor = selected.orientation === "vertical" ? "#DC2626" : "#00A36C";
    for (const segment of selected.segments) {
      overlays.push(rectSvg({
        x1: segment.bboxInImage[0],
        y1: segment.bboxInImage[1],
        x2: segment.bboxInImage[2],
        y2: segment.bboxInImage[3],
        color: segmentColor,
        strokeWidth: 2,
        dash: "6 4",
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

function polygonSvg({ points, color, strokeWidth }) {
  const pointText = points.map(([x, y]) => `${x},${y}`).join(" ");
  return `<polygon points="${pointText}" fill="none" stroke="${color}" stroke-width="${strokeWidth}"/>`;
}

function labelSvg({ x, y, text, color }) {
  const safeText = escapeXml(text);
  const labelY = Math.max(18, y - 4);
  const width = Math.min(640, Math.max(40, countCodePoints(text) * 8 + 8));
  return `<rect x="${x}" y="${labelY - 16}" width="${width}" height="20" fill="${color}" fill-opacity="0.9"/>
<text x="${x + 4}" y="${labelY - 3}" fill="#ffffff" font-size="12" font-family="sans-serif">${safeText}</text>`;
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

async function readRecognitionConfig(ymlPath) {
  const config = yaml.load(await fs.readFile(ymlPath, "utf8"));
  const transforms = config?.PreProcess?.transform_ops ?? [];
  const decodeImage = findTransform(transforms, "DecodeImage") ?? {};
  const resize = findTransform(transforms, "RecResizeImg") ?? {};
  const normalize = findTransform(transforms, "NormalizeImage") ?? {};
  const postprocess = config?.PostProcess ?? {};
  const imageShape = (resize.image_shape ?? [3, 48, 320]).map(Number);
  const channels = imageShape[0] ?? 3;

  return {
    imgMode: String(decodeImage.img_mode ?? "BGR").toUpperCase(),
    imageShape,
    scale: parseScale(normalize.scale ?? "1./255."),
    mean: normalize.mean ?? new Array(channels).fill(0.5),
    std: normalize.std ?? new Array(channels).fill(0.5),
    postprocessName: String(postprocess.name ?? "CTCLabelDecode"),
    characterDict: (postprocess.character_dict ?? []).map(String),
  };
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

function serializeImageDetectedBox(box) {
  return {
    index: box.index,
    score: round(box.score, 6),
    pointsInImage: box.pointsInImage,
    bboxInImage: box.bboxInImage.map((value) => round(value, 3)),
    center: box.center.map((value) => round(value, 3)),
    orientation: isVerticalBbox(box.bboxInImage) ? "vertical" : "horizontal",
  };
}

function serializeCandidate(candidate) {
  return {
    name: candidate.name,
    orientation: candidate.orientation,
    text: candidate.text,
    confidence: round(candidate.confidence, 6),
    charCount: candidate.charCount,
    segmentCount: candidate.segmentCount,
    textSegmentCount: candidate.textSegmentCount,
    ignoredSegmentCount: candidate.ignoredSegmentCount,
    segments: candidate.segments.map((segment) => ({
      index: segment.index,
      source: segment.source,
      detBoxIndex: segment.detBoxIndex,
      detScore: round(segment.detScore, 6),
      textRole: segment.textRole,
      ignored: segment.ignored,
      rubyAttachedTo: segment.rubyAttachedTo,
      orientation: segment.orientation,
      rotation: segment.rotation,
      bboxInCrop: segment.bboxInCrop,
      bboxInImage: segment.bboxInImage,
      pointsInCrop: segment.pointsInCrop,
      pointsInImage: segment.pointsInImage,
      text: segment.text,
      confidence: round(segment.confidence, 6),
      tokens: segment.tokens,
      cropPath: segment.cropPath,
      input: segment.input,
      outputShape: segment.outputShape,
    })),
  };
}

function normalizeImageDetectedBox(box, index, imageWidth, imageHeight) {
  const points = box.points.map(([x, y]) => [
    clamp(Number(x), 0, imageWidth),
    clamp(Number(y), 0, imageHeight),
  ]);
  const bbox = pointsToBbox(points, imageWidth, imageHeight, OCR_BOX_PADDING);
  return {
    index,
    score: box.score,
    pointsInImage: points.map((point) => point.map((value) => round(value, 3))),
    bboxInImage: bbox,
    center: boxCenter(bbox),
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

function inferCandidateOrientation(segments) {
  const verticalCount = segments.filter((segment) => segment.orientation === "vertical").length;
  const horizontalCount = segments.filter((segment) => segment.orientation === "horizontal").length;
  if (verticalCount > 0 && verticalCount >= horizontalCount) {
    return "vertical";
  }
  if (horizontalCount > 0) {
    return "horizontal";
  }
  return "unknown";
}

function classifyRubySegments(segments, bubbleOrientation) {
  if (bubbleOrientation !== "vertical") {
    return segments.map((segment) => ({
      ...segment,
      textRole: "body",
      ignored: false,
      rubyAttachedTo: null,
    }));
  }

  const verticalWidths = segments
    .filter((segment) => segment.orientation === "vertical")
    .map((segment) => boxWidth(segment.bboxInCrop));
  const bodyWidthRef = percentile(verticalWidths, 0.85);

  return segments.map((segment) => {
    const rubyTarget = findRubyTarget(segment, segments, bodyWidthRef);
    if (rubyTarget && (isKanaRubyText(segment.text) || segment.confidence < 0.5)) {
      return {
        ...segment,
        textRole: "ruby",
        ignored: true,
        rubyAttachedTo: rubyTarget.index,
      };
    }

    return {
      ...segment,
      textRole: "body",
      ignored: false,
      rubyAttachedTo: null,
    };
  });
}

function findRubyTarget(segment, segments, bodyWidthRef) {
  if (segment.orientation !== "vertical" || bodyWidthRef <= 0) {
    return null;
  }

  const box = segment.bboxInCrop;
  const width = boxWidth(box);
  if (width > bodyWidthRef * 0.72) {
    return null;
  }

  const centerX = boxCenterX(box);
  const candidates = segments
    .filter((other) => other !== segment && other.orientation === "vertical")
    .filter((other) => {
      const otherBox = other.bboxInCrop;
      const otherWidth = boxWidth(otherBox);
      if (otherWidth < width * 1.35) {
        return false;
      }

      const yOverlap = overlapRatio([box[1], box[3]], [otherBox[1], otherBox[3]]);
      if (yOverlap < 0.45) {
        return false;
      }

      const otherCenterX = boxCenterX(otherBox);
      if (centerX <= otherCenterX) {
        return false;
      }

      const horizontalGap = Math.max(0, box[0] - otherBox[2], otherBox[0] - box[2]);
      const rightSideSlack = Math.max(4, otherWidth * 0.25);
      return box[2] >= otherCenterX && box[0] <= otherBox[2] + rightSideSlack
        && horizontalGap <= Math.max(18, otherWidth * 0.8);
    })
    .sort((a, b) => {
      const aGap = Math.abs(boxCenterX(a.bboxInCrop) - centerX);
      const bGap = Math.abs(boxCenterX(b.bboxInCrop) - centerX);
      return aGap - bGap;
    });

  return candidates[0] ?? null;
}

function isKanaRubyText(text) {
  const chars = Array.from(String(text)).filter((char) => !isIgnorableRubyPunctuation(char));
  if (chars.length === 0) {
    return false;
  }

  let kanaCount = 0;
  let kanjiCount = 0;
  for (const char of chars) {
    if (isKanaLike(char)) {
      kanaCount += 1;
    }
    if (isCjkIdeograph(char)) {
      kanjiCount += 1;
    }
  }

  return kanjiCount === 0 && kanaCount > 0 && kanaCount / chars.length >= 0.75;
}

function isKanaLike(char) {
  return /[\u3041-\u3096\u309D\u309E\u30A1-\u30FA\u30FC\u30FD\u30FE]/u.test(char);
}

function isCjkIdeograph(char) {
  return /[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]/u.test(char);
}

function isIgnorableRubyPunctuation(char) {
  return /[\s。、，,.・「」『』（）()！？!?]/u.test(char);
}

function isVerticalBbox(box) {
  const width = Math.max(1, box[2] - box[0]);
  const height = Math.max(1, box[3] - box[1]);
  return height / width >= VERTICAL_ASPECT_RATIO;
}

function compareDetectedBoxReadingOrder(a, b) {
  return compareHorizontalBoxes(a, b);
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

function compareImageDetectedBoxReadingOrder(a, b) {
  const aY = a.bboxInImage[1];
  const bY = b.bboxInImage[1];
  if (Math.abs(aY - bY) > 20) {
    return aY - bY;
  }
  return a.bboxInImage[0] - b.bboxInImage[0];
}

function pointInBox(point, box) {
  return point[0] >= box[0] && point[0] <= box[2] && point[1] >= box[1] && point[1] <= box[3];
}

function offsetBox(box, offsetX, offsetY) {
  return [
    box[0] + offsetX,
    box[1] + offsetY,
    box[2] + offsetX,
    box[3] + offsetY,
  ];
}

function percentile(values, ratio) {
  if (values.length === 0) {
    return 0;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * ratio));
  return sorted[index];
}

function boxWidth(box) {
  return Math.max(1, box[2] - box[0]);
}

function boxCenterX(box) {
  return (box[0] + box[2]) / 2;
}

function boxCenter(box) {
  return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2];
}

function overlapRatio(a, b) {
  const overlap = Math.max(0, Math.min(a[1], b[1]) - Math.max(a[0], b[0]));
  const minLength = Math.max(1, Math.min(a[1] - a[0], b[1] - b[0]));
  return overlap / minLength;
}

async function readJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, "utf8"));
}

async function assertFile(filePath, makeTarget) {
  try {
    await fs.access(filePath);
  } catch {
    throw new Error(`Missing file: ${filePath}. Run "${makeTarget}" in ppocrv6 first.`);
  }
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

function normalizeComicImageSize(config) {
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

function countUniqueAssignedBoxes(textAreas) {
  const indexes = new Set();
  for (const textArea of textAreas) {
    for (const box of textArea.textDetection.boxes) {
      indexes.add(box.index);
    }
  }
  return indexes.size;
}

function weightedAverage(entries) {
  if (entries.length === 0) {
    return 0;
  }
  let totalWeight = 0;
  let total = 0;
  for (const entry of entries) {
    totalWeight += entry.weight;
    total += entry.value * entry.weight;
  }
  return totalWeight > 0 ? round(total / totalWeight, 6) : 0;
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
