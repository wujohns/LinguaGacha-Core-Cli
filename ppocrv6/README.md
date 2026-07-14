# PP-OCRv6 Comic Processing Scripts

This directory keeps only the current comic OCR and inpaint test flows.

## Scripts

```bash
npm run detect:comic:det-rec
npm run inpaint:comic:lama
npm run inpaint:comic:fill
```

All three scripts accept comma-separated input images through `INPUT_IMAGES`.
Paths may be absolute or relative to this directory.

Examples:

```bash
INPUT_IMAGES=assets/cct.png npm run detect:comic:det-rec
INPUT_IMAGES=assets/cct.png npm run inpaint:comic:lama
INPUT_IMAGES=assets/cct.png npm run inpaint:comic:fill
```

## Models

Download model files with Make:

```bash
make comic-bubble
make medium-det
make medium-rec
make lama-manga CURL_PROXY="-x http://127.0.0.1:7990"
```

Or download the grouped sets:

```bash
make models
make inpaint-models CURL_PROXY="-x http://127.0.0.1:7990"
```

## Current Flows

- `test-comic-bubble-medium-det-rec.cjs`
  - Runs `comic-bubble` once per image.
  - Runs `PP-OCRv6 medium-det` once per image.
  - Assigns whole-image detection boxes back to comic text areas.
  - Runs `PP-OCRv6 medium-rec` for recognition crops.
- `test-comic-bubble-lama-inpaint.cjs`
  - Uses `comic-bubble` text boxes as LaMa masks.
  - Keeps crop context separate from mask area.
- `test-comic-bubble-medium-fill-inpaint.cjs`
  - Runs `comic-bubble` once per image.
  - Runs `PP-OCRv6 medium-det` once per image.
  - Uses outer-ring median color fill instead of an inpaint model.
  - Defaults to `COMIC_FILL_TEXT_LABELS=text_bubble`; `text_free` is not processed unless explicitly configured.
