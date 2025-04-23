import os
import cv2
import numpy as np
from config import CONFIDENCE_THRESHOLD, MAX_BOX_AREA_RATIO, MAX_ASPECT_RATIO
import sys

sys.stdout.reconfigure(encoding='utf-8')


def filter_and_draw_boxes(image_np, boxes, scores, image_size):
    h_img, w_img = image_size
    image_area = w_img * h_img
    count = 0

    for box, score in zip(boxes, scores):
        if score < CONFIDENCE_THRESHOLD:
            continue

        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        area = width * height
        aspect_ratio = max(width / height, height / width)

        if area > MAX_BOX_AREA_RATIO * image_area or aspect_ratio > MAX_ASPECT_RATIO:
            continue

        cv2.rectangle(image_np, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        count += 1

    return image_np, count