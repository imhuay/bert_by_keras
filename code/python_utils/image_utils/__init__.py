#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-02-03 00:36
    
Author:
    huayang
    
Subject:
    处理图像相关的一些函数
"""
import os
import imghdr


def get_image_ext(image_path):
    """获取图像文件的真实后缀"""
    # 获取当前后缀
    ext_cur = os.path.splitext(image_path)[1]

    if ext_cur.startswith('.'):
        ext_cur = ext_cur[1:]

    # 获取真实后缀
    ext_real = imghdr.what(image_path)

    # 是否相同
    is_same = ext_cur == ext_real or {ext_cur, ext_real} == {'jpg', 'jpeg'}

    return ext_real, is_same


def rename_to_real_ext(image_path):
    """将图片用真实后缀重命名"""


if __name__ == '__main__':
    """"""
    image_path_ls = [
        '../data/pok.jpeg',
        '../data/pok.jpg',
        '../data/pok.png',
        '../data/test.txt',
    ]

    for path in image_path_ls:
        ext_real, is_same = get_image_ext(path)
        print(ext_real, is_same)
    """
    jpeg True
    jpeg True
    jpeg False
    None False
    """

