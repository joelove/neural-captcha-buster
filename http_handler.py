import json
import operator

from functools import reduce

import character_segmenter
import character_analyser


def read(event, context):
    base64_image = event["image"]
    letter_images = character_segmenter.get_letter_images(base64_image)
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
