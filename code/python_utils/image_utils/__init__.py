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

from PIL import Image


def get_image_ext(image_path):
    """
    获取图像文件的真实后缀
    如果不是图片，返回后缀为 None
    该方法不能判断图片是否完整

    Args:
        image_path:

    Returns:
        ext_real, is_same
        真实后缀，真实后缀与当前后缀是否相同
        如果当前文件不是图片，则 ext_real 为 None
    """
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
    """将图片重命名为真实后缀"""
    ext_real, is_same = get_image_ext(image_path)

    if is_same or ext_real is None:
        return

    prefix, _ = os.path.splitext(image_path)
    dst = '.'.join([prefix, ext_real])
    os.rename(image_path, dst)


def is_image_complete(image_path):
    """判断图片是否完整"""
    file_obj = open(image_path, 'rb')
    is_valid = True

    if 'jpeg' == imghdr.what(image_path):
        if not file_obj.read().rstrip(b'\0\r\n').endswith(b'\xff\xd9'):
            is_valid = False
    else:
        try:
            Image.open(file_obj).verify()
        except:
            is_valid = False

    return is_valid


if __name__ == '__main__':
    """"""
    dir_path = '../data'
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        ext_real, is_same = get_image_ext(file_path)
        print('%s' % '\t'.join(str(it) for it in [file_name, ext_real, is_same]))

    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        is_valid = is_image_complete(file_path)
        print('%s' % '\t'.join(str(it) for it in [file_name, is_valid]))

    # rename_to_real_ext(os.path.join(dir_path, 'pok_test.png'))
