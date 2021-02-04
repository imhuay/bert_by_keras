#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-02-04 14:34
    
Author:
    huayang
    
Subject:
    
"""
from datetime import datetime


def get_now_time(fmt="%Y%m%d%H%M%S"):
    """获取当前时间（格式化）"""
    return datetime.now().strftime(fmt)
