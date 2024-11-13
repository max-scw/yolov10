from argparse import ArgumentParser

from utils import load_yolo_from_file


if __name__ == '__main__':
    parser = ArgumentParser()
    # model
    parser.add_argument("--model-version", type=int, default=None, help="YOLO version number (3, 5, 6, 8, 9, 10)")
    parser.add_argument("--model-type", type=str, default=None, help="YOLO model type. (e.g b, l, m, n, s, x for YOLO version 10)")
    parser.add_argument("--weights", type=str, default=None, help="initial weights path")  # e.g. jameslahm/yolov10n
    # data
    parser.add_argument("--data", type=str, help="path to data file, i.e. coco128.yaml")

    args = parser.parse_args()

    model = load_yolo_from_file(args.weights, args.model_version, args.model_type)


    kwargs = {"augment": False}
    if args.data:
        kwargs = {"data": args.data}

    # load model
    model.val(**kwargs)
