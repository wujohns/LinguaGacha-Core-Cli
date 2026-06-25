#!/usr/bin/env node
const fs = require("node:fs/promises");
const path = require("node:path");

const ort = require("onnxruntime-web");
const sharp = require("sharp");
const yaml = require("js-yaml");

const SCRIPT_DIR = __dirname;
const ASSETS_DIR = path.join(SCRIPT_DIR, "assets");

const COMIC_MODEL_DIR = path.join(SCRIPT_DIR, "model", "comic-bubble");
const COMIC_MODEL_ONNX_PATH = path.join(COMIC_MODEL_DIR, "detector-v4-s_int8.onnx");
const COMIC_MODEL_CONFIG_PATH = path.join(COMIC_MODEL_DIR, "config.json");
const COMIC_PREPROCESSOR_CONFIG_PATH = path.join(COMIC_MODEL_DIR, "preprocessor_config.json");

const REC_MODEL_DIR = path.join(SCRIPT_DIR, "model", "medium-rec");
const REC_MODEL_ONNX_PATH = path.join(REC_MODEL_DIR, "inference.onnx");
const REC_MODEL_YML_PATH = path.join(REC_MODEL_DIR, "inference.yml");

const DEFAULT_INPUT_IMAGE_PATHS = [
  path.join(ASSETS_DIR, "comic_en.png"),
  path.join(ASSETS_DIR, "comic_jp.png"),
];

const INPUT_IMAGE_PATHS = (process.env.INPUT_IMAGES ?? "")
  .split(",")
  .map((entry) => entry.trim())
  .filter(Boolean)
  .map((entry) => path.resolve(SCRIPT_DIR, entry));

const COMIC_SCORE_THRESHOLD = Number(process.env.COMIC_BUBBLE_THRESHOLD ?? "0.35");
const COMIC_TEXT_LABEL = process.env.COMIC_TEXT_LABEL ?? "text_bubble";
const TEXT_CROP_PADDING = Number(process.env.COMIC_TEXT_PADDING ?? "3");
const TEXT_TRIM_PADDING = Number(process.env.COMIC_TEXT_TRIM_PADDING ?? "4");
const SEGMENT_PADDING = Number(process.env.COMIC_TEXT_SEGMENT_PADDING ?? "2");
const INK_THRESHOLD = Number(process.env.COMIC_TEXT_INK_THRESHOLD ?? "180");

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
});

async function main() {
  const inputImagePaths = INPUT_IMAGE_PATHS.length > 0 ? INPUT_IMAGE_PATHS : DEFAULT_INPUT_IMAGE_PATHS;

  const [comicModelConfig, comicPreprocessorConfig, recConfig] = await Promise.all([
    readJson(COMIC_MODEL_CONFIG_PATH),
    readJson(COMIC_PREPROCESSOR_CONFIG_PATH),
    readRecognitionConfig(REC_MODEL_YML_PATH),
    assertFile(COMIC_MODEL_ONNX_PATH, "make comic-bubble"),
    assertFile(REC_MODEL_ONNX_PATH, "make medium-rec"),
  ]);

  const comicImageSize = normalizeComicImageSize(comicPreprocessorConfig);
  const comicId2Label = normalizeId2Label(comicModelConfig.id2label);

  console.time("comic-session");
  const comicSession = await createSession(COMIC_MODEL_ONNX_PATH);
  console.timeEnd("comic-session");

  console.time("rec-session");
  const recSession = await createSession(REC_MODEL_ONNX_PATH);
  console.timeEnd("rec-session");

  const comicInputName = comicSession.inputNames.includes("images")
    ? "images"
    : comicSession.inputNames[0];
  const comicTargetSizeInputName = comicSession.inputNames.includes("orig_target_sizes")
    ? "orig_target_sizes"
    : comicSession.inputNames[1];
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
  recSession,
  recInputName,
  recOutputName,
  recConfig,
}) {
  const imageBaseName = path.basename(imagePath, path.extname(imagePath));
  const outputJsonPath = path.join(ASSETS_DIR, `${imageBaseName}.comic-bubble-medium-rec.json`);
  const debugImagePath = path.join(ASSETS_DIR, `${imageBaseName}.comic-bubble-medium-rec.png`);
  const cropDir = path.join(ASSETS_DIR, `${imageBaseName}.comic-bubble-medium-rec-crops`);

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

  console.time(`${imageBaseName}:recognition`);
  const textAreas = [];
  for (let index = 0; index < textDetections.length; index += 1) {
    const detection = textDetections[index];
    const textArea = await recognizeTextArea({
      imagePath,
      imageBaseName,
      cropDir,
      textAreaIndex: index,
      detection,
      recSession,
      recInputName,
      recOutputName,
      recConfig,
    });
    textAreas.push(textArea);
  }
  console.timeEnd(`${imageBaseName}:recognition`);

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
      recognition: {
        name: recConfig.postprocessName,
        decoder: "ctc-greedy",
        textAreaSegmentModes: ["whole", "horizontal-lines", "vertical-columns"],
      },
    },
    counts: countByLabel(allDetections),
    detections: allDetections,
    textAreas,
  };

  await fs.writeFile(outputJsonPath, `${JSON.stringify(result, null, 2)}\n`, "utf8");
  await drawDebugImage(imagePath, debugImagePath, textAreas);

  console.log(`comic detections: ${allDetections.length}`, result.counts);
  console.log(`text areas: ${textAreas.length}`);
  for (const textArea of textAreas) {
    const selected = textArea.recognition.selected;
    const text = selected.text.replace(/\s+/g, " ").trim();
    console.log(
      `  #${textArea.index + 1} ${selected.name} confidence=${selected.confidence.toFixed(3)} text=${JSON.stringify(text)}`,
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
  recSession,
  recInputName,
  recOutputName,
  recConfig,
}) {
  const crop = await cropImage(imagePath, detection.bbox, TEXT_CROP_PADDING);
  const areaCropPath = path.join(cropDir, `${imageBaseName}.text-${textAreaIndex + 1}.png`);
  await fs.writeFile(areaCropPath, crop.buffer);

  const recognitionCrop = await trimTextCrop(crop);
  const recognitionCropPath = path.join(cropDir, `${imageBaseName}.text-${textAreaIndex + 1}.trimmed.png`);
  await fs.writeFile(recognitionCropPath, recognitionCrop.buffer);

  const fullSegment = {
    index: 0,
    bboxInCrop: [0, 0, recognitionCrop.width, recognitionCrop.height],
    bboxInImage: recognitionCrop.bboxInImage,
    rotation: 0,
  };

  const [horizontalSegments, verticalSegments] = await Promise.all([
    findTextLineSegments(recognitionCrop.buffer, recognitionCrop.bboxInImage),
    findTextColumnSegments(recognitionCrop.buffer, recognitionCrop.bboxInImage),
  ]);

  const candidates = [];
  candidates.push(await recognizeCandidate({
    name: "whole",
    orientation: "unknown",
    joiner: " ",
    sourceBuffer: recognitionCrop.buffer,
    segments: [fullSegment],
    recSession,
    recInputName,
    recOutputName,
    recConfig,
  }));
  candidates.push(await recognizeCandidate({
    name: "horizontal-lines",
    orientation: "horizontal",
    joiner: "\n",
    sourceBuffer: recognitionCrop.buffer,
    segments: horizontalSegments.length > 0 ? horizontalSegments : [fullSegment],
    recSession,
    recInputName,
    recOutputName,
    recConfig,
  }));
  candidates.push(await recognizeCandidate({
    name: "vertical-columns",
    orientation: "vertical",
    joiner: "",
    sourceBuffer: recognitionCrop.buffer,
    segments: verticalSegments.length > 0 ? verticalSegments : [{ ...fullSegment, rotation: 270 }],
    recSession,
    recInputName,
    recOutputName,
    recConfig,
  }));

  const selected = chooseBestCandidate(candidates);
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
      padding: TEXT_CROP_PADDING,
      bboxInImage: crop.bboxInImage.map((value) => round(value, 3)),
      width: crop.width,
      height: crop.height,
      recognitionPath: recognitionCropPath,
      recognitionBboxInImage: recognitionCrop.bboxInImage.map((value) => round(value, 3)),
      recognitionBboxInCrop: recognitionCrop.bboxInSourceCrop.map((value) => round(value, 3)),
      recognitionWidth: recognitionCrop.width,
      recognitionHeight: recognitionCrop.height,
    },
    segmentation: {
      inkThreshold: INK_THRESHOLD,
      segmentPadding: SEGMENT_PADDING,
      trimPadding: TEXT_TRIM_PADDING,
      horizontalLineCount: horizontalSegments.length,
      verticalColumnCount: verticalSegments.length,
    },
    recognition: {
      selected: serializeCandidate(selected, true),
      candidates: candidates.map((candidate) => serializeCandidate(candidate, false)),
    },
  };
}

async function recognizeCandidate({
  name,
  orientation,
  joiner,
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

  const textSegments = recognizedSegments.filter((segment) => segment.text.length > 0);
  const text = textSegments.map((segment) => segment.text).join(joiner);
  const confidence = weightedAverage(
    textSegments.map((segment) => ({
      value: segment.confidence,
      weight: Math.max(1, countCodePoints(segment.text)),
    })),
  );
  const charCount = countCodePoints(text);
  const score = scoreCandidate({
    name,
    confidence,
    charCount,
    textSegmentCount: textSegments.length,
  });

  return {
    name,
    orientation,
    text,
    confidence,
    score,
    charCount,
    segmentCount: recognizedSegments.length,
    textSegmentCount: textSegments.length,
    segments: recognizedSegments,
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
    rotation: segment.rotation,
    bboxInCrop: segment.bboxInCrop.map((value) => round(value, 3)),
    bboxInImage: segment.bboxInImage.map((value) => round(value, 3)),
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
      `${imageBaseName}.text-${textAreaIndex + 1}.${selected.name}-${segment.index + 1}.png`,
    );
    await fs.writeFile(segmentPath, segment.imageBuffer);
    segment.cropPath = segmentPath;
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

async function trimTextCrop(crop) {
  const ink = await readInkMask(crop.buffer, { removeEdgeComponents: true });
  const centerBox = [
    Math.floor(ink.width * 0.08),
    Math.floor(ink.height * 0.08),
    Math.ceil(ink.width * 0.92),
    Math.ceil(ink.height * 0.92),
  ];
  let x1 = ink.width;
  let y1 = ink.height;
  let x2 = 0;
  let y2 = 0;

  for (let y = centerBox[1]; y < centerBox[3]; y += 1) {
    for (let x = centerBox[0]; x < centerBox[2]; x += 1) {
      if (ink.mask[y * ink.width + x]) {
        x1 = Math.min(x1, x);
        y1 = Math.min(y1, y);
        x2 = Math.max(x2, x + 1);
        y2 = Math.max(y2, y + 1);
      }
    }
  }

  if (x1 >= x2 || y1 >= y2) {
    return {
      ...crop,
      bboxInSourceCrop: [0, 0, crop.width, crop.height],
    };
  }

  const box = [
    clamp(x1 - TEXT_TRIM_PADDING, 0, crop.width),
    clamp(y1 - TEXT_TRIM_PADDING, 0, crop.height),
    clamp(x2 + TEXT_TRIM_PADDING, 0, crop.width),
    clamp(y2 + TEXT_TRIM_PADDING, 0, crop.height),
  ];
  const left = Math.round(box[0]);
  const top = Math.round(box[1]);
  const width = Math.max(1, Math.round(box[2]) - left);
  const height = Math.max(1, Math.round(box[3]) - top);
  const buffer = await sharp(crop.buffer)
    .extract({ left, top, width, height })
    .png()
    .toBuffer();

  return {
    buffer,
    width,
    height,
    bboxInSourceCrop: [left, top, left + width, top + height],
    bboxInImage: [
      crop.bboxInImage[0] + left,
      crop.bboxInImage[1] + top,
      crop.bboxInImage[0] + left + width,
      crop.bboxInImage[1] + top + height,
    ],
  };
}

async function extractSegmentBuffer(sourceBuffer, bbox, rotation) {
  const [x1, y1, x2, y2] = bbox.map((value) => Math.round(value));
  const left = Math.max(0, x1);
  const top = Math.max(0, y1);
  const width = Math.max(1, x2 - left);
  const height = Math.max(1, y2 - top);
  let image = sharp(sourceBuffer).extract({ left, top, width, height });
  if (rotation) {
    image = image.rotate(rotation);
  }
  return image.png().toBuffer();
}

async function findTextLineSegments(imageBuffer, cropBboxInImage) {
  const ink = await readInkMask(imageBuffer, { removeEdgeComponents: true });
  const rowCounts = new Array(ink.height).fill(0);
  for (let y = 0; y < ink.height; y += 1) {
    let count = 0;
    for (let x = 0; x < ink.width; x += 1) {
      if (ink.mask[y * ink.width + x]) {
        count += 1;
      }
    }
    rowCounts[y] = count;
  }

  const minInk = Math.max(2, Math.ceil(ink.width * 0.015));
  const groups = groupActiveProjection(rowCounts, minInk, Math.max(2, Math.round(ink.height * 0.01)));
  const boxes = groups
    .map((group) => projectionGroupToBox({
      ink,
      axis: "y",
      start: group.start,
      end: group.end,
      padding: SEGMENT_PADDING,
    }))
    .filter((box) => isUsefulSegmentBox(box, ink.width, ink.height));

  return boxes.map((box, index) => ({
    index,
    bboxInCrop: box,
    bboxInImage: offsetBox(box, cropBboxInImage[0], cropBboxInImage[1]),
    rotation: 0,
  }));
}

async function findTextColumnSegments(imageBuffer, cropBboxInImage) {
  const ink = await readInkMask(imageBuffer, { removeEdgeComponents: true });
  const colCounts = new Array(ink.width).fill(0);
  for (let x = 0; x < ink.width; x += 1) {
    let count = 0;
    for (let y = 0; y < ink.height; y += 1) {
      if (ink.mask[y * ink.width + x]) {
        count += 1;
      }
    }
    colCounts[x] = count;
  }

  const minInk = Math.max(2, Math.ceil(ink.height * 0.015));
  const maxGap = Math.max(3, Math.round(ink.width * 0.035));
  let groups = groupActiveProjection(colCounts, minInk, maxGap);
  if (groups.length <= 1) {
    const fallbackMinInk = Math.max(minInk + 1, Math.ceil(ink.height * 0.04));
    const fallbackGroups = groupActiveProjection(colCounts, fallbackMinInk, Math.max(2, Math.round(ink.width * 0.02)));
    if (fallbackGroups.length > groups.length) {
      groups = fallbackGroups;
    }
  }
  const boxes = groups
    .map((group) => projectionGroupToBox({
      ink,
      axis: "x",
      start: group.start,
      end: group.end,
      padding: SEGMENT_PADDING,
    }))
    .filter((box) => isUsefulSegmentBox(box, ink.width, ink.height))
    .filter((box) => isUsefulTextColumnBox(box, ink.width, ink.height))
    .sort((a, b) => b[0] - a[0] || a[1] - b[1]);

  return boxes.map((box, index) => ({
    index,
    bboxInCrop: box,
    bboxInImage: offsetBox(box, cropBboxInImage[0], cropBboxInImage[1]),
    rotation: 270,
  }));
}

async function readInkMask(imageBuffer, options = {}) {
  const { data, info } = await sharp(imageBuffer)
    .flatten({ background: "#ffffff" })
    .greyscale()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const mask = new Uint8Array(info.width * info.height);
  for (let index = 0; index < data.length; index += 1) {
    mask[index] = data[index] < INK_THRESHOLD ? 1 : 0;
  }
  if (options.removeEdgeComponents) {
    removeEdgeComponents(mask, info.width, info.height);
  }
  return {
    width: info.width,
    height: info.height,
    mask,
  };
}

function removeEdgeComponents(mask, width, height) {
  const queue = [];
  for (let x = 0; x < width; x += 1) {
    enqueueInk(queue, mask, width, x, 0);
    enqueueInk(queue, mask, width, x, height - 1);
  }
  for (let y = 0; y < height; y += 1) {
    enqueueInk(queue, mask, width, 0, y);
    enqueueInk(queue, mask, width, width - 1, y);
  }

  for (let cursor = 0; cursor < queue.length; cursor += 1) {
    const index = queue[cursor];
    const x = index % width;
    const y = Math.floor(index / width);
    if (x > 0) {
      enqueueInk(queue, mask, width, x - 1, y);
    }
    if (x + 1 < width) {
      enqueueInk(queue, mask, width, x + 1, y);
    }
    if (y > 0) {
      enqueueInk(queue, mask, width, x, y - 1);
    }
    if (y + 1 < height) {
      enqueueInk(queue, mask, width, x, y + 1);
    }
  }
}

function enqueueInk(queue, mask, width, x, y) {
  const index = y * width + x;
  if (mask[index]) {
    mask[index] = 0;
    queue.push(index);
  }
}

function groupActiveProjection(counts, minInk, maxGap) {
  const groups = [];
  let start = -1;
  let end = -1;
  let gap = 0;

  for (let index = 0; index < counts.length; index += 1) {
    const active = counts[index] >= minInk;
    if (active) {
      if (start < 0) {
        start = index;
      }
      end = index;
      gap = 0;
      continue;
    }

    if (start >= 0) {
      gap += 1;
      if (gap > maxGap) {
        groups.push({ start, end });
        start = -1;
        end = -1;
        gap = 0;
      }
    }
  }

  if (start >= 0) {
    groups.push({ start, end });
  }

  return groups;
}

function projectionGroupToBox({ ink, axis, start, end, padding }) {
  let x1 = ink.width;
  let y1 = ink.height;
  let x2 = 0;
  let y2 = 0;

  if (axis === "y") {
    for (let y = start; y <= end; y += 1) {
      for (let x = 0; x < ink.width; x += 1) {
        if (ink.mask[y * ink.width + x]) {
          x1 = Math.min(x1, x);
          y1 = Math.min(y1, y);
          x2 = Math.max(x2, x + 1);
          y2 = Math.max(y2, y + 1);
        }
      }
    }
  } else {
    for (let x = start; x <= end; x += 1) {
      for (let y = 0; y < ink.height; y += 1) {
        if (ink.mask[y * ink.width + x]) {
          x1 = Math.min(x1, x);
          y1 = Math.min(y1, y);
          x2 = Math.max(x2, x + 1);
          y2 = Math.max(y2, y + 1);
        }
      }
    }
  }

  if (x1 > x2 || y1 > y2) {
    return [0, 0, 0, 0];
  }

  return [
    clamp(x1 - padding, 0, ink.width),
    clamp(y1 - padding, 0, ink.height),
    clamp(x2 + padding, 0, ink.width),
    clamp(y2 + padding, 0, ink.height),
  ];
}

function isUsefulSegmentBox(box, imageWidth, imageHeight) {
  const width = box[2] - box[0];
  const height = box[3] - box[1];
  if (width < 5 || height < 5) {
    return false;
  }
  if (width > imageWidth * 0.98 && height > imageHeight * 0.98) {
    return false;
  }
  return true;
}

function isUsefulTextColumnBox(box, imageWidth, imageHeight) {
  const width = box[2] - box[0];
  const height = box[3] - box[1];
  if (width < Math.max(10, imageWidth * 0.08) && height < imageHeight * 0.72) {
    return false;
  }
  return true;
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
    const label = `${textArea.index + 1} ${selected.name} ${selected.confidence.toFixed(2)} ${selected.text}`
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

    const segmentColor = selected.orientation === "vertical" ? "#D97706" : "#00A36C";
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

function labelSvg({ x, y, text, color }) {
  const safeText = escapeXml(text);
  const labelY = Math.max(18, y - 4);
  const width = Math.min(520, Math.max(40, countCodePoints(text) * 8 + 8));
  return `<rect x="${x}" y="${labelY - 16}" width="${width}" height="20" fill="${color}" fill-opacity="0.9"/>
<text x="${x + 4}" y="${labelY - 3}" fill="#ffffff" font-size="12" font-family="sans-serif">${safeText}</text>`;
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

function chooseBestCandidate(candidates) {
  return [...candidates].sort((a, b) => b.score - a.score || b.confidence - a.confidence)[0];
}

function scoreCandidate({ name, confidence, charCount, textSegmentCount }) {
  if (charCount === 0 || textSegmentCount === 0) {
    return 0;
  }

  const lengthFactor = 0.75 + 0.25 * Math.min(1, charCount / 4);
  const segmentBonus = Math.min(0.08, textSegmentCount * 0.02);
  const wholePenalty = name === "whole" ? 0.03 : 0;
  return round(confidence * lengthFactor + segmentBonus - wholePenalty, 6);
}

function serializeCandidate(candidate, includeSegments) {
  const serialized = {
    name: candidate.name,
    orientation: candidate.orientation,
    text: candidate.text,
    confidence: round(candidate.confidence, 6),
    score: round(candidate.score, 6),
    charCount: candidate.charCount,
    segmentCount: candidate.segmentCount,
    textSegmentCount: candidate.textSegmentCount,
  };

  if (includeSegments) {
    serialized.segments = candidate.segments.map((segment) => ({
      index: segment.index,
      rotation: segment.rotation,
      bboxInCrop: segment.bboxInCrop,
      bboxInImage: segment.bboxInImage,
      text: segment.text,
      confidence: round(segment.confidence, 6),
      tokens: segment.tokens,
      cropPath: segment.cropPath,
      input: segment.input,
      outputShape: segment.outputShape,
    }));
  }

  return serialized;
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
