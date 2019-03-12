import json
import base64
import numpy
import cv2
import character_segmenter


def read(event, context):
    image_string = event["image"]
    characters = character_segmenter.read(image_string)

    body = {
        "image": characters
    }

    response = {
        "statusCode": 200,
        "body": json.dumps(body)
    }

    return response
