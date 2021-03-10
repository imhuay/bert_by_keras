#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-03-08 20:45
    
Author:
    huayang
    
Subject:
    
"""

import os


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
    import imghdr

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


if __name__ == '__main__':
    """"""
    dir_path = '../_test_data/'
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        if os.path.isdir(file_path):
            continue
        ext_real, is_same = get_image_ext(file_path)
        print('%s' % '\t'.join(str(it) for it in [file_name, ext_real, is_same]))

    # rename_to_real_ext(os.path.join(dir_path, 'pok_test.png'))
