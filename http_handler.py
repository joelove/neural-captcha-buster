import json
import operator
import base64

from io import BytesIO
from PIL import Image
from functools import reduce

import character_segmenter
import character_analyser


def decode_base64(base64_string):
    return Image.open(BytesIO(base64.b64decode(base64_string)))


def read(event, context):
    base64_image = event["image"]
    raw_image = decode_base64(base64_image)
    letter_images = character_segmenter.get_letter_images(raw_image)
    letter_characters = character_analyser.read_characters(letter_images)
    letters = reduce(operator.add, letter_characters)

    body = {
        "result": letters
    }

    response = {
        "statusCode": 200,
        "body": json.dumps(body)
    }

    return response
