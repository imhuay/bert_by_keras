#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-02-03 11:47
    
Author:
    huayang
    
Subject:
    常用的工具函数
"""


def get_now_time(fmt="%Y%m%d%H%M%S"):
    """获取当前时间（格式化）"""
    from datetime import datetime
    return datetime.now().strftime(fmt)
