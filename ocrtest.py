from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

from array import array
import os
from PIL import Image
import sys
import time

'''
Authenticate
Authenticates your credentials and creates a client.
'''
subscription_key = '0eaf0d01a5ee493b94a5e07f75c22cdf'#os.environ["VISION_KEY"]
endpoint = "http://ms-ocr-service:5000"#"https://15821-read.cognitiveservices.azure.com/"#os.environ["VISION_ENDPOINT"]
computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))
'''
END - Authenticate
'''

'''
OCR: Read File using the Read API, extract text - remote
This example will extract text in an image, then print results, line by line.
This API call can also extract handwriting style text (not show Key ide
# Path to local image
local_image_path = "./test.jpg"

# Open the image
with open(local_image_path, "rb") as image_stream:
    # Call API with image and raw response (allows you to get the operation location)
    read_response = computervision_client.read_in_stream(image_stream, raw=True)

# Get the operation location (URL with an ID at the end) from the response
read_operation_location = read_response.headers["Operation-Location"]
# Grab the ID from the URL
operation_id = read_operation_location.split("/")[-1]

# Call the "GET" API and wait for it to retrieve the results 
while True:
    read_result = computervision_client.get_read_result(operation_id)
    if read_result.status not in ['notStarted', 'running']:
        break
    time.sleep(0.2)

# Print the detected text, line by line
if read_result.status == OperationStatusCodes.succeeded:
    # from IPython import embed; embed()
    for text_result in read_result.analyze_result.read_results:
        for line in text_result.lines:
            print(line.text)
            # print(line.bounding_box)
print()
'''
END - Read File - remote
'''

print("End of Computer Vision quickstart.")