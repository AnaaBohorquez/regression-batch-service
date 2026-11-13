"""Command line entry point: `python -m regressor train ...` and `python -m regressor predict ...`."""

import argparse
import logging
import sys

from regressor.config import DEFAULT_CONFIG_PATH, load_config
from regressor.data import DataValidationError
from regressor.inference import predict
from regressor.training import DEFAULT_ARTIFACTS_DIR, train

logger = logging.getLogger("regressor")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="regressor", description="Batch regression pipeline.")
    commands = parser.add_subparsers(dest="command", required=True)

    train_cmd = commands.add_parser("train", help="train the model and save it to the artifacts folder")
    train_cmd.add_argument("--data", required=True, help="CSV with the features and the target column")
    train_cmd.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="model config (default: %(default)s)")
    train_cmd.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS_DIR), help="output folder (default: %(default)s)")

    predict_cmd = commands.add_parser("predict", help="predict a CSV batch with the trained model")
    predict_cmd.add_argument("--input", required=True, help="CSV with the features to predict")
    predict_cmd.add_argument("--output", required=True, help="where to write the predictions CSV")
    predict_cmd.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS_DIR), help="folder with the trained model (default: %(default)s)")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    try:
        if args.command == "train":
            metadata = train(args.data, load_config(args.config), args.artifacts)
            logger.info(
                "model trained on %d rows with %d features, saved to %s/",
                metadata["n_rows"], len(metadata["features"]), args.artifacts,
            )
        else:
            predictions = predict(args.input, args.output, args.artifacts)
            logger.info("%d predictions written to %s", len(predictions), args.output)
    except (FileNotFoundError, ValueError, DataValidationError) as err:
        logger.error(str(err))
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
