#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-02-04 14:35
    
Author:
    huayang
    
Subject:
    
"""
import platform


def get_system():
    """获取当前系统类型"""
    return platform.system()


def system_is(sys_name):
    """"""
    return get_system() == sys_name


def is_linux():
    """判断是否为 linux 系统"""
    return system_is('Linux')


def is_windows():
    """判断是否为 windows 系统"""
    return system_is('Windows')


def is_mac():
    """判断是否为 mac os 系统"""
    return system_is('Darwin')

