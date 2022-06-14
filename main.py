# -*- coding: utf-8 -*-

'''
===============================================================================
||   Authors   | 劉啟迪(Qidi Liu)
||-------------|---------------------------------------------------------------
||   License   | Private
||-------------|---------------------------------------------------------------
|| Description | 用傳統算法的組合實現穩健的影片穩定
===============================================================================
'''

__author__ = 'QidiLiu'

import cv2
import os
import tqdm
from cv2 import CV_16S
from cv2 import BORDER_DEFAULT
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('utils')

INPUT_VIDEO = 'test.avi'
GRID_M = 6
GRID_N = 8
GRID_HEIGHT = 0
GRID_WIDTH = 0
HALF_SLIDING_WINDOW_SIZE = 2
MAX_MOVING_RANGE = 30
GRID_KSIZE = 3
BLUR_KSIZE = 5


def opencv_debug(showing_image):
    while True:
        cv2.imshow('Test for debugging', showing_image)
        if (cv2.waitKey(0) == 27):
            break


def compareImages(last_gray_image, gray_image):
    gray_image = cv2.GaussianBlur(gray_image, (BLUR_KSIZE, BLUR_KSIZE), 0)
    x_grad = cv2.Sobel(gray_image, CV_16S, 1, 0, ksize=GRID_KSIZE, borderType=BORDER_DEFAULT)
    y_grad = cv2.Sobel(gray_image, CV_16S, 0, 1, ksize=GRID_KSIZE, borderType=BORDER_DEFAULT)
    abs_x_grad = cv2.convertScaleAbs(x_grad)
    abs_y_grad = cv2.convertScaleAbs(y_grad)
    grad = cv2.addWeighted(abs_x_grad, 0.5, abs_y_grad, 0.5, 0)
    opencv_debug(grad)
    best_vectors = np.zeros([GRID_M, GRID_N, 3], np.float64)

    for i in range(GRID_M):
        for j in range(GRID_N):
            y_1 = i * GRID_HEIGHT
            y_2 = (i+1) * GRID_HEIGHT
            x_1 = j * GRID_WIDTH
            x_2 = (j+1) * GRID_WIDTH
            roi = grad[y_1:y_2, x_1:x_2]
            best_vectors[i, j, 0] = np.max(roi)
            grid_max_position = np.unravel_index(np.argmax(roi), roi.shape)
            best_vectors[i, j, 1] = grid_max_position[1]
            best_vectors[i, j, 2] = grid_max_position[0]

    padding_size = HALF_SLIDING_WINDOW_SIZE + MAX_MOVING_RANGE
    padded_gray_image = cv2.copyMakeBorder(
        gray_image,
        padding_size,
        padding_size,
        padding_size,
        padding_size,
        cv2.BORDER_REPLICATE
    )
    return best_vectors


def main():
    global GRID_HEIGHT, GRID_WIDTH
    _capture = cv2.VideoCapture(INPUT_VIDEO)
    _frames_num = int(_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    _frames_h = int(_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    GRID_HEIGHT = int(_frames_h / GRID_M)
    _frames_w = int(_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    GRID_WIDTH = int(_frames_w / GRID_N)

    for frame_i in tqdm.tqdm(range(_frames_num)):
        _ret, _image = _capture.read()
        if (frame_i == 0):
            _gray_image = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
        else:
            _last_gray_image = _gray_image
            _gray_image = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
            best_vectors = compareImages(_last_gray_image, _gray_image)

            for i in tqdm.tqdm(range(GRID_M)):
                for j in range(GRID_N):
                    y_0 = i * GRID_HEIGHT
                    marker_y = round(y_0 + best_vectors[i, j, 2])
                    x_0 = j * GRID_WIDTH
                    marker_x = round(x_0 + best_vectors[i, j, 1])
                    _image = cv2.drawMarker(_image, [marker_x, marker_y], (0, 0, 255))
    
            opencv_debug(_image)


if __name__ == '__main__':
    main()
