#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-01-31 17:43
    
Author:
    huayang
    
Subject:
    
"""

from tensorflow_data.get_dataset_from_directory import get_dataset_from_directory
from tensorflow_data.utils import ds_image_show


if __name__ == '__main__':
    """"""
    dir_path = '/Users/huayang/workspace/meituan/beauty_theme/data/data_pic'
    ds_train, ds_val = get_dataset_from_directory(dir_path, label_mode='ca')
    ds_image_show(ds_train, batch_size=32, label_mode='ca')
