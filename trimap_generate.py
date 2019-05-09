#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/9 12:37
# @Author  : yanggang

"""
this use mouse event and find contours to generate trimap for image matting
"""

import cv2 as cv
import numpy as np


def draw_circle(event, x, y, flags, param):
    if flags == cv.EVENT_FLAG_LBUTTON:
        cv.circle(img, (x, y), 10, (255, 0, 0, 0), -1)
        cv.circle(img_mask, (x, y), 10, (255, 255, 255), -1)


def draw_edge(image):
    cv.namedWindow('image', cv.WINDOW_NORMAL)
    cv.resizeWindow('image', 256, 385)
    cv.setMouseCallback('image', draw_circle)
    while True:
        cv.imshow('image', img)
        if cv.waitKey(1) & 0xFF == ord('q'):  # press q to exit
            cv.imwrite("mask.png", img_mask)
            break
    cv.destroyAllWindows()
    return img_mask.astype(np.uint8)


def contours_fill(image):
    _, binImg = cv.threshold(image, 100, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(binImg, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(image, contours, 0, (255, 255, 255), cv.FILLED)
    cv.imshow('Contours Image', image)
    cv.imwrite('alpha.png', image)
    print(contours[1].shape)
    cv.waitKey()
    cv.destroyAllWindows()
    return image.astype(np.uint8)


def generate_trimap(alpha):
    fg = np.array(np.equal(alpha, 255).astype(np.float32))
    # fg = cv.erode(fg, kernel, iterations=np.random.randint(1, 3))
    unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
    unknown = cv.dilate(unknown, kernel, iterations=np.random.randint(1, 100))
    trimap = fg * 255 + (unknown - fg) * 128
    cv.imwrite('trimap.png', trimap)
    return trimap.astype(np.uint8)


def main():
    drawed_edge = draw_edge(img)
    alpha = contours_fill(drawed_edge)
    trimap = generate_trimap(alpha)
    cv.imshow('img', trimap)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    img = cv.imread('test.png')
    h, w, _ = img.shape
    img_mask = np.zeros((h, w), dtype=np.uint8)
    main()
