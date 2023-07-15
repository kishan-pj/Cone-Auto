# from __future__ import division

# import cv2
# import numpy as np

# _ORANGE = 'orange'
# _GREEN = 'green'

# _COLORS = {_ORANGE: (0, 165, 255), _GREEN: (0, 255, 0)}

# _HSV_COLOR_RANGES = {
#     _ORANGE: (np.array([0, 50, 50], dtype=np.uint8), np.array([15, 255, 255], dtype=np.uint8)),
#     _GREEN: (np.array([46, 50, 50], dtype=np.uint8), np.array([75, 255, 255], dtype=np.uint8))
# }


# def predominant_rgb_color(img, ymin, xmin, ymax, xmax):
#     crop = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[ymin:ymax, xmin:xmax]
#     best_color, highest_pxl_count = None, -1
#     for color, r in _HSV_COLOR_RANGES.items():
#         lower, upper = r
#         pxl_count = np.count_nonzero(cv2.inRange(crop, lower, upper))
#         if pxl_count > highest_pxl_count:
#             best_color = color
#             highest_pxl_count = pxl_count
#     return _COLORS[best_color]


# def add_rectangle_with_text(image, ymin, xmin, ymax, xmax, color, text):
#     cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 5)
#     cv2.putText(image, text, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2,
#                 cv2.LINE_AA)


# def resize_width_keeping_aspect_ratio(image, desired_width, interpolation=cv2.INTER_AREA):
#     (h, w) = image.shape[:2]
#     r = desired_width / w
#     dim = (desired_width, int(h * r))
#     return cv2.resize(image, dim, interpolation=interpolation)
from __future__ import division

import cv2
import numpy as np

_ORANGE = 'orange'
_GREEN = 'green'

_COLORS = {_ORANGE: (0, 165, 255), _GREEN: (0, 255, 0)}

_HSV_COLOR_RANGES = {
    _ORANGE: (np.array([0, 50, 50], dtype=np.uint8), np.array([15, 255, 255], dtype=np.uint8)),
    _GREEN: (np.array([46, 50, 50], dtype=np.uint8), np.array([75, 255, 255], dtype=np.uint8))
}

_COLOR_NAMES = {_COLORS[color]: color for color in _COLORS}


def get_color_name(rgb_color):
    return _COLOR_NAMES.get(tuple(rgb_color), 'unknown')


def predominant_rgb_color(img, ymin, xmin, ymax, xmax):
    crop = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[ymin:ymax, xmin:xmax]
    best_color, highest_pxl_count = None, -1
    for color, r in _HSV_COLOR_RANGES.items():
        lower, upper = r
        pxl_count = np.count_nonzero(cv2.inRange(crop, lower, upper))
        if pxl_count > highest_pxl_count:
            best_color = color
            highest_pxl_count = pxl_count
    return _COLORS[best_color]


def add_rectangle_with_text(image, ymin, xmin, ymax, xmax, color, text):
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 5)
    color_name = get_color_name(color)
    text = f"{color_name}: {text}"
    cv2.putText(image, text, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2,
                cv2.LINE_AA)


def resize_width_keeping_aspect_ratio(image, desired_width, interpolation=cv2.INTER_AREA):
    (h, w) = image.shape[:2]
    r = desired_width / w
    dim = (desired_width, int(h * r))
    return cv2.resize(image, dim, interpolation=interpolation)
