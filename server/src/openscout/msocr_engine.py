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



# for ms ocr dependencies
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from array import array
import os
import sys
import time

# for other dependencies
import logging
import time
import requests
import json
import cv2
import importlib_resources
import numpy as np
from gabriel_protocol import gabriel_pb2
from gabriel_server import cognitive_engine
from pathlib import Path
from PIL import Image, ImageDraw
from io import BytesIO

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
class MSOCREngine(cognitive_engine.Engine):
    ENGINE_NAME = "openscout-msocr"

    def __init__(self, args):
        # llm
        self.endpoint_llm = "http://openllm-service:5000"
        
        # ocr
        subscription_key = '0eaf0d01a5ee493b94a5e07f75c22cdf'#os.environ["VISION_KEY"]
        self.endpoint_ocr = "http://ms-ocr-service:5000"#"https://15821-read.cognitiveservices.azure.com/"#os.environ["VISION_ENDPOINT"]
        self.computervision_client = ComputerVisionClient(self.endpoint_ocr, CognitiveServicesCredentials(subscription_key))
        
        # store
        self.store_detections = args.store
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

    def infer(self, processed_res):
        headers = {"content-type": "application/json"}
        # send http request with image and receive response
        response = requests.post(
            "{}/{}".format(self.endpoint_llm, "generate"), json=processed_res, headers=headers
        )
        logger.info(f"debug: {type(response)}, {dir(response)}, {response}")
        return response.text
    
    def msocr_process(self, img):
        logger.info("===== Read File - remote =====")
        # Call API with image and raw response (allows you to get the operation location)
        read_response = self.computervision_client.read_in_stream(img, raw=True)

        # Get the operation location (URL with an ID at the end) from the response
        read_operation_location = read_response.headers["Operation-Location"]
        # Grab the ID from the URL
        operation_id = read_operation_location.split("/")[-1]

        # Call the "GET" API and wait for it to retrieve the results 
        while True:
            read_result = self.computervision_client.get_read_result(operation_id)
            if read_result.status not in ['notStarted', 'running']:
                break
            time.sleep(0.2)
            
        # Initialize an empty list to collect lines of text
        detected_lines = []

        # Print the detected text, line by line
        if read_result.status == OperationStatusCodes.succeeded:
            # from IPython import embed; embed()
            for text_result in read_result.analyze_result.read_results:
                for line in text_result.lines:
                    detected_lines.append(line.text)

        # Concatenate all lines of detected text into one string with a newline as the separator
        concatenated_text = '\n'.join(detected_lines)
        logger.info(concatenated_text)
        logger.info("End of Computer Vision quickstart.")
        return concatenated_text
        
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
        
        # configure time
        self.t0 = time.time()
        
        # process image
        np_data = np.fromstring(input_frame.payloads[0], dtype=np.uint8)
        image_np = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(image_np) #PIL image
        byte_stream = BytesIO()
        img.save(byte_stream, format='JPEG')  # Use 'JPEG' or another format as needed
        byte_stream.seek(0)  # Rewind the stream to the beginning
        processed_result = self.msocr_process(byte_stream)
        
        # infer
        # processed_result = f"I am in front of this sign, what should I do? Sign:{processed_result}"
        processed_result = f"Write one sentence in English about the key idea of below text :{processed_result} Key idea in English: "
        infer_pack = {"text": processed_result}
        logger.info(f"infer_pack : {infer_pack}")
        inferred_result = self.infer(infer_pack)
        logger.info(f"inferred : {inferred_result}")
        
        # combine processed and inferred result
        results = f"Transcribed : {processed_result} \n" + f"inferred : {inferred_result}\n"
            
        # configure result wrappper
        status = gabriel_pb2.ResultWrapper.Status.SUCCESS
        result_wrapper = cognitive_engine.create_result_wrapper(status)
        result_wrapper.result_producer_name.value = self.ENGINE_NAME

        if self.store_detections:
            timestamp_millis = int(time.time() * 1000)
            filename = str(timestamp_millis) + ".jpg"
            path = self.storage_path / "received" / filename
            img.save(path, format="JPEG")

        if len(results) > 0:
            result = gabriel_pb2.ResultWrapper.Result()
            result.payload_type = gabriel_pb2.PayloadType.TEXT
            result.payload = results.encode(encoding="utf-8")
            result_wrapper.results.append(result)

            if self.store_detections:
                try:
                    filename = str(timestamp_millis) + ".txt"
                    path = self.storage_path / "transcribed" / filename
                    with open(path, 'w') as file:
                        file.write(results)
                    logger.info(f"Stored transcript: {path}")
                except Exception as e:  # Catch other potential exceptions
                    print(f"An unexpected error occurred: {e}")
                    return result_wrapper

        return result_wrapper