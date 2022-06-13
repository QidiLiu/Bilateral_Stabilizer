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
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('utils')

INPUT_VIDEO = 'test.avi'


def main():
    _capture = cv2.VideoCapture(INPUT_VIDEO)
    _frames_num = int(_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    _frames_h = int(_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    _frames_w = int(_capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    for frame_i in range(_frames_num):
        ret, image = _capture.read()


if __name__ == '__main__':
    main()
