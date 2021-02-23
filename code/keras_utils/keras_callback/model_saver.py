#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-01-26 19:33
    
Author:
    huayang
    
Subject:
    
"""
import os
import json
import datetime

try:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
except:
    import keras
    import keras.backend as K


class ModelSaver(keras.callbacks.Callback):
    """保存最佳模型 Callback

    References:
        keras.callbacks.ModelCheckpoint

    Examples:
        metric = 'accuracy'
        callbacks = [
            ModelSaver('./best_model', metric=metric, metric_expect=0.9),
        ]
        model.compile(optimizer='sgd', loss="binary_crossentropy", metrics=[metric])
        model.fit(ds_train, epochs=epochs, validation_data=ds_val, verbose=1, callbacks=callbacks)
    """

    def __init__(self,
                 save_path='./best_model',
                 save_log_path=None,
                 metric='accuracy',
                 metric_delta=0.02,
                 metric_expect=0.9,
                 is_early_stop=True,
                 min_delta=0.003,
                 best_keep_count=4):
        """
        
        Args:
            save_path: 保存路径，默认为 './best_model'
            save_log_path: 模型保存日志的路径，日志为 json 格式，默认为 os.path.join(save_path, 'save_log.json')
            metric: 参考指标（训练集），keras 中验证集对应指标为 'val_%s' % metric
            metric_delta: 训练集与验证集之间的最大差值，只有两者的差值在这个之间才会保存，防止保存过拟合的模型
            metric_expect: 最小期望指标，只有验证集指标大于该值才会保存
            is_early_stop: 提前停止训练，停止条件：最优指标保持一定次数
            min_delta: 允许最优指标发生的波动
            best_keep_count: 最优指标需要保持的次数
        """
        super(ModelSaver, self).__init__()
        self.save_path = save_path
        self.save_log_file_path = save_log_path if save_log_path else os.path.join(save_path, 'save_log.json')
        self.save_log = [] if not os.path.exists(self.save_log_file_path) else json.load(open(self.save_log_file_path))
        self.metric = metric
        self.metric_val = 'val_%s' % metric
        self.metric_delta = metric_delta
        self.train_time = datetime.datetime.now().strftime('%Y.%m.%d-%H:%M')

        self.metric_best_last = None
        if len(self.save_log) > 0:
            epoch_log_best = self.save_log[-1]
            self.metric_best_last = epoch_log_best[self.metric_val]
            metric_expect = self.metric_best_last if self.metric_best_last > metric_expect else metric_expect

        self.metric_expect = metric_expect
        self.best = metric_expect
        self.is_early_stop = is_early_stop
        self.min_delta = min_delta
        self.best_keep_count = best_keep_count
        self.early_stop_count = 0

    def on_train_begin(self, logs=None):
        """"""
        if self.metric_best_last:
            print('Last best %s=%.3f, so replace the metric_expect with it.'
                  % (self.metric_val, self.metric_best_last))

    def on_epoch_end(self, epoch, logs=None):
        """"""
        metric = logs.get(self.metric)
        metric_val = logs.get(self.metric_val)

        if metric_val > self.best and abs(metric - metric_val) < self.metric_delta:
            self.model.save(self.save_path, overwrite=True)
            self.best = metric_val

            epoch_log = {
                'train_time': self.train_time,
                'epoch': epoch,
                self.metric: metric,
                self.metric_val: metric_val,
            }
            self.save_log.append(epoch_log)
            json.dump(self.save_log, 
                      open(self.save_log_file_path, 'w'), ensure_ascii=False, indent=2)
            print('Save model at epoch %s with %s=%.3f and %s=%.3f.'
                  % (epoch, self.metric, metric, self.metric_val, metric_val))

        # 连续一定次数保持在 best 范围内
        if abs(metric_val - self.best) < self.min_delta:
            self.early_stop_count += 1
        else:
            self.early_stop_count = 0

        if self.is_early_stop and self.early_stop_count >= self.best_keep_count:
            self.model.stop_training = True
            print('Early stop train model at epoch %s with %s=%.3f and %s=%.3f.'
                  % (epoch, self.metric, metric, self.metric_val, metric_val))
