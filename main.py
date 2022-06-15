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
MAX_SHIFT = 30
GRID_KSIZE = 3
BLUR_KSIZE = 5


def opencv_debug(showing_image):
    while True:
        cv2.imshow('Test for debugging', showing_image)
        if (cv2.waitKey(0) == 27):
            break


def compareImages(last_gray_image, gray_image):
    last_gray_image = cv2.GaussianBlur(last_gray_image, (BLUR_KSIZE, BLUR_KSIZE), 0)
    x_grad = cv2.Sobel(last_gray_image, cv2.CV_16S, 1, 0, ksize=GRID_KSIZE, borderType=cv2.BORDER_DEFAULT)
    y_grad = cv2.Sobel(last_gray_image, cv2.CV_16S, 0, 1, ksize=GRID_KSIZE, borderType=cv2.BORDER_DEFAULT)
    abs_x_grad = cv2.convertScaleAbs(x_grad)
    abs_y_grad = cv2.convertScaleAbs(y_grad)
    grad = cv2.addWeighted(abs_x_grad, 0.5, abs_y_grad, 0.5, 0)
    best_vectors = np.zeros([GRID_M, GRID_N, 3], np.float64)
    shifts_num = (1 + 2*MAX_SHIFT) ** 2
    crop_roi_size = 1 + 2*HALF_SLIDING_WINDOW_SIZE
    shifted_rois = np.zeros(
        [GRID_M*crop_roi_size, GRID_N*crop_roi_size, shifts_num],
        np.uint8
    )
    broadcasted_comparing_rois = np.zeros(
        [GRID_M*crop_roi_size, GRID_N*crop_roi_size, shifts_num],
        np.uint8
    )
    comparing_rois = np.zeros(
        [GRID_M*crop_roi_size, GRID_N*crop_roi_size],
        np.uint8
    )
    padding_size = HALF_SLIDING_WINDOW_SIZE + MAX_SHIFT
    padded_gray_image = cv2.copyMakeBorder(
        gray_image,
        padding_size,
        padding_size,
        padding_size,
        padding_size,
        cv2.BORDER_REPLICATE
    )
    padded_last_gray_image = cv2.copyMakeBorder(
        last_gray_image,
        padding_size,
        padding_size,
        padding_size,
        padding_size,
        cv2.BORDER_REPLICATE
    )

    shift_count = 0
    for shift_i in range(1 + 2*MAX_SHIFT):
        for shift_j in range(1 + 2*MAX_SHIFT):
            for i in range(GRID_M):
                for j in range(GRID_N):
                    y_1 = i * GRID_HEIGHT
                    y_2 = (i+1) * GRID_HEIGHT
                    x_1 = j * GRID_WIDTH
                    x_2 = (j+1) * GRID_WIDTH
                    roi = grad[y_1:y_2, x_1:x_2]
                    grid_max_position = np.unravel_index(np.argmax(roi), roi.shape) # format: (y, x)
                    grid_y_0 = padding_size + i*GRID_WIDTH
                    grid_x_0 = padding_size + j*GRID_WIDTH
                    reference_y = grid_y_0 + grid_max_position[0] - HALF_SLIDING_WINDOW_SIZE
                    reference_x = grid_x_0 + grid_max_position[1] - HALF_SLIDING_WINDOW_SIZE
                    crop_y_1 = i*crop_roi_size
                    crop_y_2 = (i+1)*crop_roi_size
                    crop_x_1 = j*crop_roi_size
                    crop_x_2 = (j+1)*crop_roi_size
                    if (shift_i == 0 and shift_j == 0):
                        comparing_rois[crop_y_1:crop_y_2, crop_x_1:crop_x_2] = padded_last_gray_image[
                            reference_y : reference_y + crop_roi_size,
                            reference_x : reference_x + crop_roi_size
                        ]
                        #opencv_debug(comparing_rois)
                    shift_reference_y = reference_y - MAX_SHIFT
                    shift_reference_x = reference_x - MAX_SHIFT
                    shift_y_1 = shift_reference_y + shift_i
                    shift_y_2 = shift_reference_y + shift_i + crop_roi_size
                    shift_x_1 = shift_reference_x + shift_j
                    shift_x_2 = shift_reference_x + shift_j + crop_roi_size
                    shifted_rois[crop_y_1:crop_y_2, crop_x_1:crop_x_2, shift_count] = padded_gray_image[
                        shift_y_1:shift_y_2,
                        shift_x_1:shift_x_2
                    ]
            shift_count += 1
    
    comparing_rois = np.expand_dims(comparing_rois, axis=-1)
    broadcasted_comparing_rois += comparing_rois
    roi_abs_diffs = np.sum(cv2.absdiff(broadcasted_comparing_rois, shifted_rois), (0, 1))
    diffs_image = np.reshape(roi_abs_diffs, [1 + 2*MAX_SHIFT, 1 + 2*MAX_SHIFT])
    print(diffs_image.shape)
    print(np.argmin(diffs_image))
    print(np.min(diffs_image))
    opencv_debug(diffs_image/100000)

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


if __name__ == '__main__':
    main()
