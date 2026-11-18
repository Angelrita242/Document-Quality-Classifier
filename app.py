import tempfile
from pathlib import Path

import gradio as gr
from pipeline import QualityPipeline
import spaces

@spaces.GPU
def zero_gpu_check():
    return None

pipeline = QualityPipeline()

def check_document(image):
    if image is None:
        return {"status": "error", "reason": "Please upload an image."}

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
        image_path = Path(temp.name)

    try:
        image.save(image_path, format="PNG")
        return pipeline.predict(str(image_path))
    except Exception as exc:
        return {"status": "error", "reason": str(exc)}
    finally:
        image_path.unlink(missing_ok=True)

demo = gr.Interface(
    fn=check_document,
    inputs=gr.Image(type="pil", label="Upload document image"),
    outputs=gr.JSON(label="Quality assessment"),
    title="Document Quality Checker",
    description="Upload an image to check whether its quality is acceptable."
)

if __name__ == "__main__":
    demo.launch()
