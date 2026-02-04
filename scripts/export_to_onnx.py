import torch
import torchvision.models as models

# Load pretrained model (example: ResNet backbone)
model = models.resnet18(pretrained=True)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "models/onnx/detector.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11,
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
)

print("ONNX export successful")
