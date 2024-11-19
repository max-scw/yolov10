from utils import (
    load_yolo_from_file,
    build_argument_parser_from_yaml_file,
    add_arguments_build_model,
    parse_arguments_defaults
)


if __name__ == '__main__':
    parser, config = build_argument_parser_from_yaml_file()
    # model
    parser = add_arguments_build_model(parser)

    args, logger = parse_arguments_defaults(parser)



    model = load_yolo_from_file(args.weights, args.model_version, args.model_type)

    if args.augment:
        logger.info("Augmentation disabled to evaluate the test data.")
        args.augment = False
    if not args.device:
        logger.info("Augmentation disabled to evaluate the test data.")
        args.device = "cpu"


    # load model
    model.val(**{ky: vl for ky, vl in args.__dict__.items() if ky in config.keys()})
