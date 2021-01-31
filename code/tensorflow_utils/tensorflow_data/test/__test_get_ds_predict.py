#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-01-31 16:22
    
Author:
    huayang
    
Subject:
    
"""

try:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
except:
    import keras
    import keras.backend as K

from tensorflow_data.get_ds_predict import get_ds_predict_image
from tensorflow_data.utils import ds_image_show

if __name__ == '__main__':
    """"""
    data_path = r'/Users/huayang/workspace/meituan/beauty_theme/data/data_test/20210129/imgs'
    batch_size = 32
    ds_predict, file_ls, file_name_ls = get_ds_predict_image(data_path, batch_size=batch_size)
    ds_image_show(ds_predict, file_name_ls, with_label=False, n_show=12, batch_size=batch_size)

