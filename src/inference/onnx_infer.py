import onnxruntime as ort
import numpy as np

def run_onnx_inference():
    session = ort.InferenceSession(
        "models/onnx/detector.onnx",
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

    output = session.run(None, {input_name: dummy_input})
    return output

if __name__ == "__main__":
    run_onnx_inference()
    print("ONNX Runtime inference completed.")
