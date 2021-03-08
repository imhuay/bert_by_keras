#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-03-01 17:10
    
Author:
    huayang
    
Subject:
    视频转图像（抽帧）
"""

import os
import cv2


def video2image(video_path, save_dir, n_frame=None, n_step=None):
    """
    视频抽帧

    Args:
        video_path: 视频路径
        save_dir: 图像保存文件夹
        n_frame: 按固定帧数抽帧
        n_step: 按固定间隔抽帧

    Returns:
        None
    """
    assert n_frame is None or n_step is None, '不同时设置 n_frame 和 n_step'
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(fps)

    frame_ls = []
    for _ in range(fps):
        ret, frame = cap.read()
        if ret:
            frame_ls.append(frame)

    if n_frame is not None:
        n_step = len(frame_ls) // n_frame + 1 if len(frame_ls) % n_frame != 0 else len(frame_ls) // n_frame

    final_frame_ls = frame_ls[::n_step] if n_step is not None and n_step > 0 else frame_ls

    for frame_id, frame in enumerate(final_frame_ls):
        cv2.imwrite(os.path.join(save_dir, '%.04d.jpg' % (frame_id + 1)), frame)


if __name__ == '__main__':
    """"""
    video_path = r'../_test_data/v_ApplyEyeMakeup_g01_c01.avi'
    save_dir = r'../_test_data/out'
    video2image(video_path, save_dir, n_step=10)
