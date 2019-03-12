import json
import base64
import numpy
import cv2
import character_segmenter
import base64
from io import StringIO
from PIL import Image


def read_base64(base64_string):
    sbuf = StringIO()
    sbuf.write(base64.b64decode(base64_string))
    pimg = Image.open(sbuf)
    return cv2.cvtColor(np.array(pimg), cv2.COLOR_GRAY2RGB)


def read(event, context):
    base64_image = event["image"]
    image_buffer = read_base64(base64_image)
    characters = character_segmenter.read(image_buffer)

    body = {
        "image": characters
    }

    response = {
        "statusCode": 200,
        "body": json.dumps(body)
    }

    return response
