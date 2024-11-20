from utils import (
    load_yolo_from_file,
    build_argument_parser_from_yaml_file,
    add_arguments_build_model,
    parse_arguments_defaults
)



if __name__ == "__main__":
    parser, config = build_argument_parser_from_yaml_file()
    # model
    parser = add_arguments_build_model(parser)

    args, logger = parse_arguments_defaults(parser)

    model = load_yolo_from_file(args.weights, args.model_version, args.model_type, args.task)

    # Train the model
    model.train(
        **{ky: vl for ky, vl in args.__dict__.items() if ky in config.keys()}
    )
