
# Model Card: ConvNeXt-Tiny Document Quality Classifier

## Model overview

The model classifies document images into good and bad quality
categories before downstream OCR processing.

## Architecture

Fine-tuned ConvNeXt-Tiny exported to ONNX.

## Input

- RGB image
- 224 × 224 pixels
- ImageNet normalization

## Output

One logit converted to a bad-quality probability using sigmoid.

## Class mapping

- Good = 0
- Bad = 1

## Decision threshold

0.2, based on the selected Task 5B threshold.

## Intended use

- Document image quality screening.
- Identifying images that may require resubmission.
- Reducing poor-quality images entering downstream OCR.

## Limitations

- Binary classification does not identify the specific defect.
- Performance may vary across document types and capture devices.
- The model requires further validation on representative real-world
  documents.
- Document-type classification is not implemented in this version.

## Responsible use

The model should support quality screening rather than act as the sole
authority for important decisions. A resubmission or human-review
process should remain available.

Document images may contain sensitive personal information and should
be processed securely.
