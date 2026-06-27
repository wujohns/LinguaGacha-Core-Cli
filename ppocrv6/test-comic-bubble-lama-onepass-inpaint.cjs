#!/usr/bin/env node
const fs = require("node:fs/promises");
const path = require("node:path");

const ort = require("onnxruntime-web");
const sharp = require("sharp");

const SCRIPT_DIR = __dirname;
const ASSETS_DIR = path.join(SCRIPT_DIR, "assets");

const COMIC_MODEL_DIR = path.join(SCRIPT_DIR, "model", "comic-bubble");
const COMIC_MODEL_ONNX_PATH = path.join(COMIC_MODEL_DIR, "detector-v4-s_int8.onnx");
const COMIC_MODEL_CONFIG_PATH = path.join(COMIC_MODEL_DIR, "config.json");
const COMIC_PREPROCESSOR_CONFIG_PATH = path.join(COMIC_MODEL_DIR, "preprocessor_config.json");

const LAMA_MODEL_DIR = path.join(SCRIPT_DIR, "model", "lama-manga");
const LAMA_MODEL_ONNX_PATH = path.join(LAMA_MODEL_DIR, "lama-manga.onnx");
const LAMA_SIZE = 512;

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
const MASK_LABELS = readStringListEnv("COMIC_LAMA_MASK_LABELS", ["text_bubble", "text_free"]);
const CONTEXT_LABELS = readStringListEnv("COMIC_LAMA_CONTEXT_LABELS", ["bubble"]);
const MASK_BOX_PADDING = readNumberEnv("COMIC_LAMA_MASK_PADDING", 0);
const CONTEXT_PADDING = readNumberEnv("COMIC_LAMA_CONTEXT_PADDING", 24);
const MAX_MASKS = readNumberEnv("COMIC_LAMA_LIMIT", 0);
const MIN_MASK_AREA = readNumberEnv("COMIC_LAMA_MIN_MASK_AREA", 16);
const ONEPASS_CROP_MODE = readEnumEnv("COMIC_LAMA_ONEPASS_CROP", "context", ["context", "mask", "page"]);

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

  await Promise.all([
    assertFile(COMIC_MODEL_ONNX_PATH, "make comic-bubble"),
    assertFile(LAMA_MODEL_ONNX_PATH, 'make lama-manga CURL_PROXY="-x http://127.0.0.1:7990"'),
  ]);

  const [comicModelConfig, comicPreprocessorConfig] = await Promise.all([
    readJson(COMIC_MODEL_CONFIG_PATH),
    readJson(COMIC_PREPROCESSOR_CONFIG_PATH),
  ]);

  const comicImageSize = normalizeComicImageSize(comicPreprocessorConfig);
  const comicId2Label = normalizeId2Label(comicModelConfig.id2label);

  console.time("comic-session");
  const comicSession = await createSession(COMIC_MODEL_ONNX_PATH);
  console.timeEnd("comic-session");

  console.time("lama-session");
  const lamaSession = await createSession(LAMA_MODEL_ONNX_PATH);
  console.timeEnd("lama-session");

  validateLamaSession(lamaSession);

  const comicInputName = comicSession.inputNames.includes("images")
    ? "images"
    : comicSession.inputNames[0];
  const comicTargetSizeInputName = comicSession.inputNames.includes("orig_target_sizes")
    ? "orig_target_sizes"
    : comicSession.inputNames[1];

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
      lamaSession,
    });
  }
}

async function createSession(modelPath) {
  const modelBytes = await fs.readFile(modelPath);
  return ort.InferenceSession.create(modelBytes, {
    executionProviders: ["wasm"],
  });
}

function validateLamaSession(session) {
  const missingInputs = ["image", "mask"].filter((name) => !session.inputNames.includes(name));
  if (missingInputs.length > 0 || !session.outputNames.includes("output")) {
    throw new Error(
      `Unexpected LaMa ONNX IO. inputs=${session.inputNames.join(", ")} outputs=${session.outputNames.join(", ")}`,
    );
  }
}

function printHelp() {
  console.log(`Usage:
  npm run inpaint:comic:lama:onepass
  INPUT_IMAGES=assets/cctt.png npm run inpaint:comic:lama:onepass

Models:
  make comic-bubble
  make lama-manga CURL_PROXY="-x http://127.0.0.1:7990"

Environment:
  INPUT_IMAGES                 Comma-separated image paths, relative to ppocrv6/ or absolute.
  COMIC_BUBBLE_THRESHOLD       Comic detector score threshold. Default: ${COMIC_SCORE_THRESHOLD}
  COMIC_LAMA_MASK_LABELS       Detector labels added into the one-pass mask. Default: ${MASK_LABELS.join(",")}
  COMIC_LAMA_CONTEXT_LABELS    Detector labels used when building the one-pass crop. Default: ${CONTEXT_LABELS.join(",")}
  COMIC_LAMA_MASK_PADDING      Pixels added around each detector frame before masking. Default: ${MASK_BOX_PADDING}
  COMIC_LAMA_CONTEXT_PADDING   Pixels added around the one-pass crop. Default: ${CONTEXT_PADDING}
  COMIC_LAMA_LIMIT             Max masks included; 0 means all. Default: ${MAX_MASKS}
  COMIC_LAMA_MIN_MASK_AREA     Skip masks smaller than this pixel area. Default: ${MIN_MASK_AREA}
  COMIC_LAMA_ONEPASS_CROP      context | mask | page. Default: ${ONEPASS_CROP_MODE}
`);
}

async function runImage({
  imagePath,
  comicSession,
  comicInputName,
  comicTargetSizeInputName,
  comicImageSize,
  comicId2Label,
  comicPreprocessorConfig,
  lamaSession,
}) {
  const imageBaseName = path.basename(imagePath, path.extname(imagePath));
  const outputJsonPath = path.join(ASSETS_DIR, `${imageBaseName}.comic-lama-onepass-inpaint.json`);
  const outputImagePath = path.join(ASSETS_DIR, `${imageBaseName}.comic-lama-onepass-clean.png`);
  const maskImagePath = path.join(ASSETS_DIR, `${imageBaseName}.comic-lama-onepass-mask.png`);
  const debugImagePath = path.join(ASSETS_DIR, `${imageBaseName}.comic-lama-onepass-debug.png`);
  const outputDir = path.join(ASSETS_DIR, `${imageBaseName}.comic-lama-onepass`);

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

  const detections = decodeComicDetections(
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

  const maskLabelSet = new Set(MASK_LABELS);
  const contextLabelSet = new Set(CONTEXT_LABELS);
  const contextDetections = detections.filter((detection) => contextLabelSet.has(detection.label));
  const maskDetections = detections
    .filter((detection) => maskLabelSet.has(detection.label))
    .sort(compareDetectionReadingOrder);
  const selectedDetections = MAX_MASKS > 0
    ? maskDetections.slice(0, MAX_MASKS)
    : maskDetections;

  await fs.mkdir(outputDir, { recursive: true });

  const metadata = await sharp(imagePath).metadata();
  if (!metadata.width || !metadata.height) {
    throw new Error(`Unable to read image dimensions: ${imagePath}`);
  }

  const { data: imageData } = await sharp(imagePath)
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const outputData = Buffer.from(imageData);
  const pageMask = Buffer.alloc(metadata.width * metadata.height, 0);

  const masks = selectedDetections.map((detection, index) => buildMaskPlan({
    index,
    detection,
    contextDetections,
    imageWidth: metadata.width,
    imageHeight: metadata.height,
  }));
  for (const mask of masks) {
    if (bboxArea(mask.maskBox) < MIN_MASK_AREA) {
      mask.skipped = true;
      mask.skipReason = "mask area below threshold";
      continue;
    }
    fillMaskRect(pageMask, metadata.width, metadata.height, mask.maskBox, 255);
  }
  const activeMasks = masks.filter((mask) => !mask.skipped);
  const globalCropBox = activeMasks.length > 0
    ? buildOnePassCropBox({
      masks: activeMasks,
      imageWidth: metadata.width,
      imageHeight: metadata.height,
    })
    : null;

  let onePass = {
    lamaRuns: 0,
    cropBox: globalCropBox,
    cropCleanImage: null,
    cropMask: null,
  };

  if (activeMasks.length > 0) {
    console.log(
      `  one-pass masks=${activeMasks.length} crop=${formatBox(globalCropBox)} mode=${ONEPASS_CROP_MODE}`,
    );

    console.time(`${imageBaseName}:lama-onepass`);
    const inpainted = await runLamaOnePass({
      imagePath,
      lamaSession,
      cropBox: globalCropBox,
      maskBoxes: activeMasks.map((mask) => mask.maskBox),
      outputDir,
      imageBaseName,
    });
    console.timeEnd(`${imageBaseName}:lama-onepass`);

    blendMaskedCropIntoImage({
      outputData,
      imageWidth: metadata.width,
      cropBox: globalCropBox,
      maskBoxes: activeMasks.map((mask) => mask.maskBox),
      cropData: inpainted.cropData,
      cropWidth: inpainted.cropWidth,
      cropHeight: inpainted.cropHeight,
    });

    onePass = {
      lamaRuns: 1,
      cropBox: globalCropBox,
      cropCleanImage: inpainted.cropCleanImagePath,
      cropMask: inpainted.cropMaskPath,
    };
  } else {
    console.log("  one-pass skipped: no active masks");
  }

  await sharp(outputData, {
    raw: {
      width: metadata.width,
      height: metadata.height,
      channels: 3,
    },
  }).png().toFile(outputImagePath);

  await sharp(pageMask, {
    raw: {
      width: metadata.width,
      height: metadata.height,
      channels: 1,
    },
  }).png().toFile(maskImagePath);

  await drawDebugImage({
    imagePath,
    outputPath: debugImagePath,
    detections,
    masks,
    globalCropBox,
    imageWidth: metadata.width,
    imageHeight: metadata.height,
  });

  const result = {
    image: path.resolve(imagePath),
    outputs: {
      cleanedImage: outputImagePath,
      mask: maskImagePath,
      debug: debugImagePath,
      workDir: outputDir,
      cropCleanImage: onePass.cropCleanImage,
      cropMask: onePass.cropMask,
    },
    models: {
      comicBubble: {
        repository: "ogkalu/comic-text-and-bubble-detector",
        variant: "detector-v4-s_int8.onnx",
        onnx: COMIC_MODEL_ONNX_PATH,
        config: COMIC_MODEL_CONFIG_PATH,
        preprocessor: COMIC_PREPROCESSOR_CONFIG_PATH,
      },
      lama: {
        repository: "mayocream/lama-manga-onnx",
        variant: "lama-manga.onnx",
        onnx: LAMA_MODEL_ONNX_PATH,
        inputShape: [1, 3, LAMA_SIZE, LAMA_SIZE],
        maskShape: [1, 1, LAMA_SIZE, LAMA_SIZE],
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
      lama: {
        inputNames: lamaSession.inputNames,
        outputNames: lamaSession.outputNames,
      },
    },
    settings: {
      comicScoreThreshold: COMIC_SCORE_THRESHOLD,
      maskLabels: MASK_LABELS,
      contextLabels: CONTEXT_LABELS,
      maskBoxPadding: MASK_BOX_PADDING,
      contextPadding: CONTEXT_PADDING,
      maxMasks: MAX_MASKS,
      minMaskArea: MIN_MASK_AREA,
      onePassCropMode: ONEPASS_CROP_MODE,
      lamaSize: LAMA_SIZE,
    },
    counts: {
      detections: countByLabel(detections),
      maskCandidates: maskDetections.length,
      selectedMasks: masks.length,
      processedMasks: activeMasks.length,
      skippedMasks: masks.filter((mask) => mask.skipped).length,
      lamaRuns: onePass.lamaRuns,
    },
    onePass,
    detections,
    masks: masks.map(serializeMask),
  };

  await fs.writeFile(outputJsonPath, `${JSON.stringify(result, null, 2)}\n`, "utf8");

  console.log(`detections: ${detections.length}`, result.counts.detections);
  console.log(`mask candidates: ${maskDetections.length}; processed masks: ${activeMasks.length}; lama runs: ${onePass.lamaRuns}`);
  console.log(`clean: ${path.relative(SCRIPT_DIR, outputImagePath)}`);
  console.log(`mask: ${path.relative(SCRIPT_DIR, maskImagePath)}`);
  console.log(`debug: ${path.relative(SCRIPT_DIR, debugImagePath)}`);
  console.log(`json: ${path.relative(SCRIPT_DIR, outputJsonPath)}`);
}

function buildMaskPlan({ index, detection, contextDetections, imageWidth, imageHeight }) {
  const sourceBox = integerBox(detection.bbox, imageWidth, imageHeight);
  const maskBox = integerBox(expandBox(detection.bbox, MASK_BOX_PADDING), imageWidth, imageHeight);
  const context = findBestContextDetection(detection, contextDetections);
  return {
    index,
    detectionIndex: detection.index,
    detectionLabel: detection.label,
    detectionScore: detection.score,
    contextDetectionIndex: context?.index ?? null,
    contextDetectionLabel: context?.label ?? null,
    contextBox: context ? integerBox(context.bbox, imageWidth, imageHeight) : null,
    sourceBox,
    maskBox,
    skipped: false,
    skipReason: null,
  };
}

function buildOnePassCropBox({ masks, imageWidth, imageHeight }) {
  if (ONEPASS_CROP_MODE === "page") {
    return [0, 0, imageWidth, imageHeight];
  }

  let box = masks[0].maskBox;
  for (const mask of masks.slice(1)) {
    box = unionBoxes(box, mask.maskBox);
  }

  if (ONEPASS_CROP_MODE === "context") {
    for (const mask of masks) {
      if (mask.contextBox) {
        box = unionBoxes(box, mask.contextBox);
      }
    }
  }

  return makeSquareBox(expandBox(box, CONTEXT_PADDING), imageWidth, imageHeight);
}

async function runLamaOnePass({
  imagePath,
  lamaSession,
  cropBox,
  maskBoxes,
  outputDir,
  imageBaseName,
}) {
  const cropWidth = cropBox[2] - cropBox[0];
  const cropHeight = cropBox[3] - cropBox[1];
  const extracted = await sharp(imagePath)
    .extract({
      left: cropBox[0],
      top: cropBox[1],
      width: cropWidth,
      height: cropHeight,
    })
    .resize(LAMA_SIZE, LAMA_SIZE, { fit: "fill" })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const maskData = createLamaMask(cropBox, maskBoxes);
  const imageTensorData = rgbBufferToCHWFloat(extracted.data, LAMA_SIZE, LAMA_SIZE);
  const outputs = await lamaSession.run({
    image: new ort.Tensor("float32", imageTensorData, [1, 3, LAMA_SIZE, LAMA_SIZE]),
    mask: new ort.Tensor("float32", maskData, [1, 1, LAMA_SIZE, LAMA_SIZE]),
  });
  const outputTensor = outputs.output;
  if (!outputTensor) {
    throw new Error(`LaMa output not found. Available outputs: ${Object.keys(outputs).join(", ")}`);
  }

  const lamaRgb = chwFloatToRgbBuffer(outputTensor.data, LAMA_SIZE, LAMA_SIZE);
  const resizedCrop = await sharp(lamaRgb, {
    raw: {
      width: LAMA_SIZE,
      height: LAMA_SIZE,
      channels: 3,
    },
  })
    .resize(cropWidth, cropHeight, { fit: "fill" })
    .raw()
    .toBuffer();

  const cropCleanImagePath = path.join(outputDir, `${imageBaseName}.lama-onepass-crop-clean.png`);
  const cropMaskPath = path.join(outputDir, `${imageBaseName}.lama-onepass-crop-mask.png`);
  await Promise.all([
    sharp(resizedCrop, {
      raw: {
        width: cropWidth,
        height: cropHeight,
        channels: 3,
      },
    }).png().toFile(cropCleanImagePath),
    writeCropMaskImage({ cropBox, maskBoxes, outputPath: cropMaskPath }),
  ]);

  return {
    cropData: resizedCrop,
    cropWidth,
    cropHeight,
    cropCleanImagePath,
    cropMaskPath,
  };
}

function createLamaMask(cropBox, maskBoxes) {
  const mask = new Float32Array(LAMA_SIZE * LAMA_SIZE);
  const cropWidth = cropBox[2] - cropBox[0];
  const cropHeight = cropBox[3] - cropBox[1];
  for (const maskBox of maskBoxes) {
    const x1 = clamp(Math.floor(((maskBox[0] - cropBox[0]) / cropWidth) * LAMA_SIZE), 0, LAMA_SIZE);
    const y1 = clamp(Math.floor(((maskBox[1] - cropBox[1]) / cropHeight) * LAMA_SIZE), 0, LAMA_SIZE);
    const x2 = clamp(Math.ceil(((maskBox[2] - cropBox[0]) / cropWidth) * LAMA_SIZE), 0, LAMA_SIZE);
    const y2 = clamp(Math.ceil(((maskBox[3] - cropBox[1]) / cropHeight) * LAMA_SIZE), 0, LAMA_SIZE);
    for (let y = y1; y < y2; y += 1) {
      mask.fill(1, y * LAMA_SIZE + x1, y * LAMA_SIZE + x2);
    }
  }
  return mask;
}

async function writeCropMaskImage({ cropBox, maskBoxes, outputPath }) {
  const cropWidth = cropBox[2] - cropBox[0];
  const cropHeight = cropBox[3] - cropBox[1];
  const mask = Buffer.alloc(cropWidth * cropHeight, 0);
  for (const maskBox of maskBoxes) {
    fillMaskRect(mask, cropWidth, cropHeight, [
      maskBox[0] - cropBox[0],
      maskBox[1] - cropBox[1],
      maskBox[2] - cropBox[0],
      maskBox[3] - cropBox[1],
    ], 255);
  }
  await sharp(mask, {
    raw: {
      width: cropWidth,
      height: cropHeight,
      channels: 1,
    },
  }).png().toFile(outputPath);
}

function blendMaskedCropIntoImage({
  outputData,
  imageWidth,
  cropBox,
  maskBoxes,
  cropData,
  cropWidth,
  cropHeight,
}) {
  for (const maskBox of maskBoxes) {
    const localMaskBox = [
      clamp(maskBox[0] - cropBox[0], 0, cropWidth),
      clamp(maskBox[1] - cropBox[1], 0, cropHeight),
      clamp(maskBox[2] - cropBox[0], 0, cropWidth),
      clamp(maskBox[3] - cropBox[1], 0, cropHeight),
    ];
    for (let y = localMaskBox[1]; y < localMaskBox[3]; y += 1) {
      const sourceRow = y * cropWidth * 3;
      const targetRow = ((cropBox[1] + y) * imageWidth + cropBox[0]) * 3;
      for (let x = localMaskBox[0]; x < localMaskBox[2]; x += 1) {
        const sourceIndex = sourceRow + x * 3;
        const targetIndex = targetRow + x * 3;
        outputData[targetIndex] = cropData[sourceIndex];
        outputData[targetIndex + 1] = cropData[sourceIndex + 1];
        outputData[targetIndex + 2] = cropData[sourceIndex + 2];
      }
    }
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

function findBestContextDetection(detection, contextDetections) {
  if (contextDetections.length === 0) {
    return null;
  }
  const center = bboxCenter(detection.bbox);
  let best = null;
  let bestScore = -Infinity;
  for (const context of contextDetections) {
    const contains = pointInBox(center, context.bbox);
    const overlap = iou(detection.bbox, context.bbox);
    const score = (contains ? 1 : 0) + overlap;
    if (score > bestScore) {
      best = context;
      bestScore = score;
    }
  }
  return bestScore > 0 ? best : null;
}

function rgbBufferToCHWFloat(buffer, width, height) {
  const planeSize = width * height;
  const tensor = new Float32Array(3 * planeSize);
  for (let index = 0; index < planeSize; index += 1) {
    tensor[index] = buffer[index * 3] / 255;
    tensor[planeSize + index] = buffer[index * 3 + 1] / 255;
    tensor[planeSize * 2 + index] = buffer[index * 3 + 2] / 255;
  }
  return tensor;
}

function chwFloatToRgbBuffer(data, width, height) {
  const planeSize = width * height;
  const buffer = Buffer.alloc(planeSize * 3);
  for (let index = 0; index < planeSize; index += 1) {
    buffer[index * 3] = floatToByte(data[index]);
    buffer[index * 3 + 1] = floatToByte(data[planeSize + index]);
    buffer[index * 3 + 2] = floatToByte(data[planeSize * 2 + index]);
  }
  return buffer;
}

function floatToByte(value) {
  return clamp(Math.round(Number(value) * 255), 0, 255);
}

async function drawDebugImage({ imagePath, outputPath, detections, masks, globalCropBox, imageWidth, imageHeight }) {
  const overlays = [];
  for (const detection of detections) {
    overlays.push(rectSvg({
      box: detection.bbox,
      color: labelColor(detection.label),
      strokeWidth: 2,
      fillOpacity: 0,
    }));
  }
  if (globalCropBox) {
    overlays.push(rectSvg({
      box: globalCropBox,
      color: "#2563EB",
      strokeWidth: 2,
      dash: "7 5",
      fillOpacity: 0,
    }));
  }
  for (const mask of masks) {
    if (mask.skipped) {
      continue;
    }
    overlays.push(rectSvg({
      box: mask.maskBox,
      color: "#DC2626",
      strokeWidth: 2,
      fillOpacity: 0.25,
    }));
    overlays.push(labelSvg({
      x: mask.maskBox[0],
      y: mask.maskBox[1],
      text: `#${mask.index + 1} ${mask.detectionLabel} ${mask.detectionScore.toFixed(2)}`,
      color: "#DC2626",
    }));
  }
  const svg = `<svg width="${imageWidth}" height="${imageHeight}" viewBox="0 0 ${imageWidth} ${imageHeight}" xmlns="http://www.w3.org/2000/svg">${overlays.join("")}</svg>`;
  await sharp(imagePath).composite([{ input: Buffer.from(svg), top: 0, left: 0 }]).toFile(outputPath);
}

function rectSvg({ box, color, strokeWidth, fillOpacity, dash }) {
  const [x1, y1, x2, y2] = box;
  const dashAttribute = dash ? ` stroke-dasharray="${dash}"` : "";
  const fill = fillOpacity ? ` fill="${color}" fill-opacity="${fillOpacity}"` : ' fill="none"';
  return `<rect x="${x1}" y="${y1}" width="${Math.max(0, x2 - x1)}" height="${Math.max(0, y2 - y1)}"${fill} stroke="${color}" stroke-width="${strokeWidth}"${dashAttribute}/>`;
}

function labelSvg({ x, y, text, color }) {
  const safeText = escapeXml(text);
  const labelY = Math.max(18, y - 4);
  const width = Math.min(420, Math.max(40, countCodePoints(text) * 8 + 8));
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

function fillMaskRect(mask, width, height, box, value) {
  const [x1, y1, x2, y2] = integerBox(box, width, height);
  for (let y = y1; y < y2; y += 1) {
    mask.fill(value, y * width + x1, y * width + x2);
  }
}

function makeSquareBox(box, imageWidth, imageHeight) {
  const integer = integerBox(box, imageWidth, imageHeight);
  const width = integer[2] - integer[0];
  const height = integer[3] - integer[1];
  const side = Math.max(width, height, 1);
  const cx = (integer[0] + integer[2]) / 2;
  const cy = (integer[1] + integer[3]) / 2;
  let left = Math.round(cx - side / 2);
  let top = Math.round(cy - side / 2);
  let right = left + side;
  let bottom = top + side;

  if (left < 0) {
    right -= left;
    left = 0;
  }
  if (top < 0) {
    bottom -= top;
    top = 0;
  }
  if (right > imageWidth) {
    left -= right - imageWidth;
    right = imageWidth;
  }
  if (bottom > imageHeight) {
    top -= bottom - imageHeight;
    bottom = imageHeight;
  }

  return [
    clamp(left, 0, imageWidth - 1),
    clamp(top, 0, imageHeight - 1),
    clamp(right, 1, imageWidth),
    clamp(bottom, 1, imageHeight),
  ];
}

function expandBox(box, padding) {
  return [
    box[0] - padding,
    box[1] - padding,
    box[2] + padding,
    box[3] + padding,
  ];
}

function integerBox(box, imageWidth, imageHeight) {
  const x1 = clamp(Math.floor(box[0]), 0, imageWidth - 1);
  const y1 = clamp(Math.floor(box[1]), 0, imageHeight - 1);
  const x2 = clamp(Math.ceil(box[2]), x1 + 1, imageWidth);
  const y2 = clamp(Math.ceil(box[3]), y1 + 1, imageHeight);
  return [x1, y1, x2, y2];
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

function pointInBox(point, box) {
  return point[0] >= box[0] && point[0] <= box[2] && point[1] >= box[1] && point[1] <= box[3];
}

function iou(a, b) {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]);
  const y2 = Math.min(a[3], b[3]);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  if (intersection === 0) {
    return 0;
  }
  return intersection / (bboxArea(a) + bboxArea(b) - intersection);
}

function compareDetectionReadingOrder(a, b) {
  const aY = a.bbox[1];
  const bY = b.bbox[1];
  if (Math.abs(aY - bY) > 20) {
    return aY - bY;
  }
  return a.bbox[0] - b.bbox[0];
}

function serializeMask(mask) {
  return {
    index: mask.index,
    detectionIndex: mask.detectionIndex,
    detectionLabel: mask.detectionLabel,
    detectionScore: mask.detectionScore,
    contextDetectionIndex: mask.contextDetectionIndex,
    contextDetectionLabel: mask.contextDetectionLabel,
    contextBox: mask.contextBox,
    sourceBox: mask.sourceBox,
    maskBox: mask.maskBox,
    skipped: Boolean(mask.skipped),
    skipReason: mask.skipReason,
  };
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

function readEnumEnv(name, fallback, choices) {
  const value = process.env[name];
  if (value == null || value.trim() === "") {
    return fallback;
  }
  const normalized = value.trim().toLowerCase();
  return choices.includes(normalized) ? normalized : fallback;
}

function formatBox(box) {
  return `[${box.map((value) => Math.round(value)).join(",")}]`;
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
