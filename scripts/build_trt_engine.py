import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def build_engine(onnx_path, engine_path, fp16=True):
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        builder.max_batch_size = 1
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB

        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        with open(onnx_path, "rb") as f:
            parser.parse(f.read())

        engine = builder.build_engine(network, config)
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())

build_engine(
    "models/onnx/detector.onnx",
    "models/tensorrt/detector_fp16.engine"
)
