import cv2.cv2 as cv2
import numpy as np
import sys

from PIL import Image
from functools import reduce
from operator import itemgetter


def debug(*objs):
    print(*objs, file=sys.stderr)


def pad_and_resize_letter(image):
    desired_size = 48
    old_size = image.size
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    resized_image = image.resize(new_size, Image.ANTIALIAS)
    new_image = Image.new("RGB", (desired_size, desired_size), (255, 255, 255))
    new_image.paste(resized_image, ((desired_size-new_size[0])//2, (desired_size-new_size[1])//2))

    return new_image


def get_letter_images(raw_image, EXPECTED_LENGTH=8):
    grayscale_image = cv2.cvtColor(np.array(raw_image), cv2.COLOR_BGR2GRAY)
    boxes = get_letter_bounding_boxes(grayscale_image, EXPECTED_LENGTH)

    def crop_letter(box):
        return raw_image.crop(box=(box[3][0]+1, box[3][1]+1, box[1][0], box[1][1]))

    letters = list(map(crop_letter, boxes))

    def get_max_letter_size(image):
        return max(image.size)

    def get_min_letter_size(letters):
        return min(list(map(get_max_letter_size, letters)))

    if not get_min_letter_size(letters):
        return None

    resized_letters = list(map(pad_and_resize_letter, letters))

    return resized_letters;


def get_letter_bounding_boxes(image, EXPECTED_LENGTH):
    _, thresh = cv2.threshold(image, 127, 255, 0)
    [contours, _] = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bbs = bounding_boxes(flatten_innermost(contours[1:]))
    merged = merge_vertical_overlaps(bbs)

    while len(merged) > EXPECTED_LENGTH:
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
    return boxes[:first_index] + [bounding_box(boxes[first_index] + boxes[first_index+1])] + boxes[first_index+2:]


def box_width(box):
    return box[1][0] - box[0][0]
