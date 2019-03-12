import json
import base64
import numpy
import cv2
import character_segmenter


def read(event, context):
    image_array = numpy.fromstring(event["image"], numpy.uint8)
    image_buffer = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    images_segments = character_segmenter.read(image_buffer)

    body = {
        "image": images_segments
    }

    response = {
        "statusCode": 200,
        "body": json.dumps(body)
    }

    return response
