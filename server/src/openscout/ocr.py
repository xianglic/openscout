#!/usr/bin/env python3
# OpenScout
#   - Distributed Automated Situational Awareness
#
#   Author: Thomas Eiszler <teiszler@andrew.cmu.edu>
#
#   Copyright (C) 2020 Carnegie Mellon University
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#
import argparse
import logging
import subprocess

from gabriel_server.network_engine import engine_runner

from .ocr_engine import OCREngine
from .timing_engine import TimingObjectEngine

SOURCE = "openscout"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--timing", action="store_true", help="Print timing information"
    )

    # parser.add_argument("-p", "--port", type=int, default=9099, help="Set port number")

    # parser.add_argument(
    #     "-m",
    #     "--model",
    #     default="coco",
    #     help=(
    #         "(OBJECT DETECTION) Subdirectory under /openscout-server/model/"
    #         " which contains Tensorflow model to load initially."
    #     ),
    # )

    parser.add_argument(
        "-r", "--threshold", type=float, default=0.85, help="Confidence threshold"
    )

    parser.add_argument(
        "-s",
        "--store",
        action="store_true",
        default=False,
        help="Store images with bounding boxes",
    )

    parser.add_argument(
        "-g",
        "--gabriel",
        default="tcp://gabriel-server:5555",
        help="Gabriel server endpoint.",
    )

    parser.add_argument(
        "-src", "--source", default=SOURCE, help="Source for engine to register with."
    )

    parser.add_argument(
        "-x",
        "--exclude",
        help=(
            "Comma separated list of classes (ids) to exclude when peforming detection."
            " Consult model/<model_name>/label_map.pbtxt."
        ),
    )

    args, _ = parser.parse_known_args()

    def ocr_engine_setup():
        if args.timing:
            engine = TimingObjectEngine(args)
        else:
            engine = OCREngine(args)

        return engine

    logger.info("Starting filebeat...")
    subprocess.call(["service", "filebeat", "start"])
    logger.info("Starting object detection cognitive engine..")
    engine_runner.run(
        engine=ocr_engine_setup(),
        source_name=args.source,
        server_address=args.gabriel,
        all_responses_required=True,
    )


if __name__ == "__main__":
    main()
