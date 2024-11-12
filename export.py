from argparse import ArgumentParser

from utils import load_yolo_from_file


if __name__ == '__main__':
    parser = ArgumentParser()
    # model
    parser.add_argument("--model-version", type=int, default=None, help="YOLO version number (3, 5, 6, 8, 9, 10)")
    parser.add_argument("--model-type", type=str, default=None, help="YOLO model type. (e.g b, l, m, n, s, x for YOLO version 10)")
    parser.add_argument("--weights", type=str, default=None, help="initial weights path")  # e.g. jameslahm/yolov10n

    # format to export the model to
    parser.add_argument("--format", type=str, default="ONNX",
                        help="Export format. For options see https://docs.ultralytics.com/modes/export/#export-formats")
    # details
    parser.add_argument("--precision", type=str, default="fp32",
                        help="Precision: FP32, FP16, INT8. Integer quantization is only available for the CoreML format.")
    parser.add_argument("--dynamic", action="store_true",
                        help="Dynamic axes. Available for the ONNX/TF/TensorRT formats.")
    # ONNX-specific flags
    parser.add_argument("--simplify", action="store_true",
                        help="Simplify model using 'onnxslim'. Only available for the ONNX format.")
    parser.add_argument("--opset", type=int, default=None, help="ONNX opset version.")


    # keras: False  # (bool) use Kera=s
    # optimize: False  # (bool) TorchScript: optimize for mobile
    # int8: False  # (bool) CoreML/TF INT8 quantization
    # dynamic: False  # (bool) ONNX/TF/TensorRT: dynamic axes
    # simplify: False  # (bool) ONNX: simplify model using `onnxslim`
    # opset:  # (int, optional) ONNX: opset version
    # workspace: 4  # (int) TensorRT: workspace size (GB)
    # nms: False  # (bool) CoreML: add NMS

    args = parser.parse_args()

    kwargs = dict()
    if args.precision.lower() == "fp32":
        pass
    elif args.precision.lower() == "fp16":
        kwargs["half"] = True
    elif args.precision.lower() == "int8":
        assert args.format.lower() == "coreml", ValueError("INT8 quantization only available for CoreML format.")
        kwargs["int8"] = True

    if args.simplify:
        assert args.format.lower() == "onnx", ValueError("--simplify only available for ONNX format.")
        kwargs["simplify"] = True

    if args.dynamic:
        assert args.format.lower() in ["onnx", "tf", "tensorrt"], ValueError(
            "--dynamic only available for ONNX/TF/TensorRT formats.")
        kwargs["dynamic"] = True

    if args.opset is not None:
        assert args.format.lower() == "onnx", ValueError("--opset only available for ONNX format.")
        kwargs["opset"] = int(args.opset)

    model = load_yolo_from_file(args.weights, args.model_version, args.model_type)

    # load model
    model.export(
        format=args.format.lower(),
        **kwargs
    )
