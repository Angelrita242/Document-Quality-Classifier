# DQC Retrospective

## What was completed

The existing fine-tuned ConvNeXt-Tiny quality classifier was exported to ONNX and integrated into a document image quality pipeline.

The pipeline performs image preprocessing, model inference and threshold-based acceptance or rejection. The classification threshold and model path are configurable through `config.yaml`.

A Gradio interface was developed to allow users to upload document images and view quality predictions.

Automated tests were implemented to check the response schema, image predictions, error handling, inference latency, threshold configuration and batch processing.

## What worked well

Reusing the existing fine-tuned ONNX model avoided unnecessary retraining and made deployment more efficient.

The configurable threshold allows the decision boundary to be adjusted without modifying or retraining the model.

Separating the quality pipeline from the user interface also makes the system easier to maintain and extend.

## Challenges

The pipeline required consistent preprocessing between model training and deployment to ensure reliable predictions.

The ONNX model also required a compatible inference runtime and correct model-path configuration.

Preparing the application for cloud deployment required adapting the original Streamlit interface to Gradio.

## Limitations

The current system only performs document image quality classification.

Document-type classification has not been implemented because it is outside the scope of this semester's work and will be developed next semester.

The deployment environment may produce different latency results from the original notebook benchmark.

## Future improvements

- Implement document-type classification.
- Validate performance on more real-world document images.
- Evaluate model quantization to reduce model size and inference time.
- Benchmark the complete deployed pipeline.
- Improve monitoring and error reporting.
- Expand automated testing with additional real-world image cases.