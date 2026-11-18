---
title: Lab 7 Document Quality Checker
emoji: 📄
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
---

# Document Image Quality Checker

A document image quality classification application powered by a fine-tuned ConvNeXt-Tiny model exported to ONNX.

## Features

- Upload document images.
- Assess image quality using the ONNX model.
- Classify images as good or bad.
- Accept or reject images using a configurable threshold.
- Return structured JSON predictions.

## Model

The application uses a fine-tuned ConvNeXt-Tiny quality classifier.

The quality threshold and model path are configured in `config.yaml`.

## Scope

This semester's implementation focuses on document image quality classification. Document-type classification is planned for the next semester.