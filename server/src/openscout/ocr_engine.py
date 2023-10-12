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

import logging
import os
import time
import pytesseract

import cv2
import importlib_resources
import numpy as np
from gabriel_protocol import gabriel_pb2
from gabriel_server import cognitive_engine
from pathlib import Path
from PIL import Image, ImageDraw
from .protocol import openscout_pb2

# setup the log
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

detection_log = logging.getLogger("ocr-engine")
fh = logging.FileHandler("/openscout-server/openscout-ocr-engine.log")
fh.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")
fh.setFormatter(formatter)
detection_log.addHandler(fh)


# class
class OCREngine(cognitive_engine.Engine):
    ENGINE_NAME = "openscout-ocr"

    def __init__(self, args):
        self.threshold = args.threshold
        self.store_detections = args.store

        if args.exclude:
            self.exclusions = list(
                map(int, args.exclude.split(","))
            )  # split string to int list
            logger.info(f"Excluding the following class ids: {self.exclusions}")
        else:
            self.exclusions = None

        logger.info(f"Confidence Threshold: {self.threshold}")

        if self.store_detections:
            watermark_path = importlib_resources.files("openscout").joinpath(
                "watermark.png"
            )
            self.watermark = Image.open(watermark_path)
            self.storage_path = Path.cwd() / "images"
            try:
                (self.storage_path / "received").mkdir(parents=True, exist_ok=True)
                (self.storage_path / "detected").mkdir(parents=True, exist_ok=True)
            except FileExistsError:
                logger.info("Images directory already exists.")
            logger.info(f"Storing detection images at {self.storage_path}")

    def handle(self, input_frame):
        
        # if the payload is TEXT, say from a CNC client, we ignore
        if input_frame.payload_type == gabriel_pb2.PayloadType.TEXT:
            status = gabriel_pb2.ResultWrapper.Status.SUCCESS
            result_wrapper = cognitive_engine.create_result_wrapper(status)
            result_wrapper.result_producer_name.value = self.ENGINE_NAME
            result = gabriel_pb2.ResultWrapper.Result()
            result.payload_type = gabriel_pb2.PayloadType.TEXT
            result.payload = "Ignoring TEXT payload.".encode(encoding="utf-8")
            result_wrapper.results.append(result)
            return result_wrapper

        extras = cognitive_engine.unpack_extras(openscout_pb2.Extras, input_frame)
                
        
        # configure time
        self.t0 = time.time()
        
        # process image
        results = pytesseract.image_to_string(input_frame.payloads[0])
        np_data = np.fromstring(input_frame.payloads[0], dtype=np.uint8)
        image_np = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        image_np = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        timestamp_millis = int(time.time() * 1000)
        
        # configure result wrappper
        status = gabriel_pb2.ResultWrapper.Status.SUCCESS
        result_wrapper = cognitive_engine.create_result_wrapper(status)
        result_wrapper.result_producer_name.value = self.ENGINE_NAME

        if self.store_detections:
            filename = str(timestamp_millis) + ".jpg"
            img = Image.fromarray(image_np)
            path = self.storage_path / "received" / filename
            img.save(path, format="JPEG")

        if len(results.pred) > 0:
            result = gabriel_pb2.ResultWrapper.Result()
            result.payload_type = gabriel_pb2.PayloadType.TEXT
            result.payload = results
            result_wrapper.results.append(result)

            if self.store_detections:
                try:
                    # results._run(
                    #     save=True,
                    #     labels=True,
                    #     save_dir=Path("openscout-vol/")
                    # )
                    results.render()
                    img = Image.fromarray(results.ims[0])
                    draw = ImageDraw.Draw(img)
                    draw.bitmap((0, 0), self.watermark, fill=None)
                    path = self.storage_path / "detected" / filename
                    img.save(path, format="JPEG")
                    logger.info(f"Stored image: {path}")
                except IndexError:
                    logger.exception("IndexError while getting bounding boxes")
                    return result_wrapper

        return result_wrapper