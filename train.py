from utils import (
    load_yolo_from_file,
    build_argument_parser_from_yaml_file,
    add_arguments_build_model,
    parse_arguments_defaults
)


def read_yaml_file(file_path) -> dict | None:
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            return data
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None


if __name__ == "__main__":
    parser, config = build_argument_parser_from_yaml_file()
    # model
    parser = add_arguments_build_model(parser)

    args, logger = parse_arguments_defaults(parser)

    # set logging level
    logger = set_logging_level(LOGGING_NAME, args.logging_level)

    logger.debug(f"Input arguments train.py: {args}")

    set_process_title(args.process_title)

    # freeze all layers up to the given layer if only one number was provided
    if len(args.freeze) == 1:
        args.freeze = list(range(0, args.freeze[0]))

    t0 = default_timer()
    default_args = read_yaml_file(Path("ultralytics/cfg/default.yaml"))

    model = load_yolo_from_file(args.weights, args.model_version, args.model_type, args.task)

    # Train the model
    model.train(
        **{ky: vl for ky, vl in args.__dict__.items() if ky in config.keys()}
    )
