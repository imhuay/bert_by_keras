#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-03-09 10:42
    
Author:
    huayang
    
Subject:
    任意对象的序列化与反序列化
"""
import base64


def byte_to_str(byte_obj, encoding='utf8'):
    """"""
    return base64.b64encode(byte_obj).decode(encoding)


def str_to_byte(byte_str, encoding='utf8'):
    return base64.b64decode(byte_str.encode(encoding))


if __name__ == '__main__':
    """"""
    # 文本文件
    obj = open('../../_test_data/test.txt', 'rb').read()
    s = byte_to_str(obj)
    obj_str = str_to_byte(s)
    with open('../../_test_data/out/test.txt', 'wb') as f:
        f.write(obj_str)

    # 图片
    obj = open('../../_test_data/pok.jpg', 'rb').read()
    s = byte_to_str(obj)
    obj_str = str_to_byte(s)
    with open('../../_test_data/out/pok.jpg', 'wb') as f:
        f.write(obj_str)

    # 视频
    obj = open('../../_test_data/v_ApplyEyeMakeup_g01_c01.avi', 'rb').read()
    s = byte_to_str(obj)
    obj_str = str_to_byte(s)
    with open('../../_test_data/out/v_ApplyEyeMakeup_g01_c01.avi', 'wb') as f:
        f.write(obj_str)
