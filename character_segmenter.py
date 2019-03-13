import cv2.cv2 as cv2
import numpy as np
import os
import sys
from functools import reduce
from operator import itemgetter
from PIL import Image



EXPECTED_MAX_CHARS = 7


def debug(*objs):
    print(*objs, file=sys.stderr)


def get_letter_images(rgb_image):
    boxes = get_letter_bounding_boxes(rgb_image)
    raw_image = Image.fromarray(rgb_image)

    def crop_letter(box):
        return raw_image.crop(
            box=(
                box[3][0]+1,
                box[3][1]+1,
                box[1][0],
                box[1][1]))

    letters = list(map(crop_letter, boxes))

    return letters;


def get_letter_bounding_boxes(image):
    imgray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(imgray, 127, 255, 0)
    [contours, _] = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bbs = bounding_boxes(flatten_innermost(contours[1:]))
    merged = merge_vertical_overlaps(bbs)

    while len(merged) > EXPECTED_MAX_CHARS:
        merged = merge_two_thinnest_adjacent_boxes(merged)

    return merged


def flatten_innermost(contours):
    return list(map(lambda contour: list(map(lambda c: c[0], contour)), contours))


def bounding_box(contour):
    x_coords = list(map(lambda coord: coord[0], contour))
    y_coords = list(map(lambda coord: coord[1], contour))
    x_min, y_min, x_max, y_max = min(x_coords), min(
        y_coords), max(x_coords), max(y_coords)

    return [[x_min, y_max], [x_max, y_max], [x_max, y_min], [x_min, y_min]]


def bounding_boxes(contours):
    return list(sorted(map(bounding_box, contours), key=by_x_min))


def by_x_min(box):
    return box[0][0]


def merge_vertical_overlaps(bounding_boxes):
    return list(reduce(merge_to_previous_if_overlap, bounding_boxes, []))


def merge_to_previous_if_overlap(merged, current_box):
    if len(merged) and has_vertical_overlap(merged[-1], current_box):
        merged[-1] = bounding_box(merged[-1] + current_box)
    else:
        merged.append(current_box)
    return merged


def has_vertical_overlap(bounding_box1, bounding_box2):
    x_min1 = bounding_box1[0][0]
    x_max1 = bounding_box1[1][0]
    x_min2 = bounding_box2[0][0]
    x_max2 = bounding_box2[1][0]
    return max(x_min1, x_min2) < min(x_max1, x_max2)


def merge_two_thinnest_adjacent_boxes(boxes):
    widths = list(map(box_width, boxes))
    adjacent_sums = [x + y for x, y in zip(widths, widths[1:] + [0])][:-1]
    first_index = min(enumerate(adjacent_sums), key=itemgetter(1))[0]
    debug(widths, adjacent_sums, first_index)
    return boxes[:first_index] + [bounding_box(boxes[first_index] + boxes[first_index+1])] + boxes[first_index+2:]


def box_width(box):
    return box[1][0] - box[0][0]
