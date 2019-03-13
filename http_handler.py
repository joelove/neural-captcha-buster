import json
import base64
import numpy
import cv2
import character_segmenter
import base64
import numpy as np
from io import BytesIO
from PIL import Image


def read_base64(base64_string):
    return Image.open(BytesIO(base64.b64decode(base64_string)))


def convert_to_rgb(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)


def read(event, context):
    base64_image = event["image"]
    raw_image = read_base64(base64_image)
    rgb_image = convert_to_rgb(raw_image)
    bounding_boxes = character_segmenter.get_letter_bounding_boxes(rgb_image)

    # Test the output
    contours = np.array(bounding_boxes, np.int32)
    test_rgb_image = rgb_image.copy()
    cv2.drawContours(test_rgb_image, contours, -1, (255, 0, 0))
    test_raw_image = Image.fromarray(test_rgb_image)
    test_raw_image.show()

    body = {
        "image": "foo"
    }

    response = {
        "statusCode": 200,
        "body": json.dumps(body)
    }

    return response
