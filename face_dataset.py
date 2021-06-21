# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import urllib.request

@tf.function
def transform_targets_for_output(y_true, grid_size, anchor_idxs):
    # y_true : (batch,n_boxes, (x1, y1, x2, y2, label, best_anchor))
    # y_true_out : (batch,grid, grid, anchors, [x1, y1, x2, y2, obj, label])

    N = tf.shape(y_true)[0]
    y_true_out = tf.zeros((N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))

    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    #
    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0

    #
    for i in tf.range(N):
        for j in tf.range(tf.shape(y_true)[1]):
            if tf.equal(y_true[i][j][2], 0):
                continue
            anchor_eq = tf.equal(anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))

            if tf.reduce_any(anchor_eq):
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2

                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                grid_xy = tf.cast(box_xy // (1/grid_size), tf.int32)

                indexes = indexes.write(idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                updates = updates.write(idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])

                idx += 1
    return tf.tensor_scatter_nd_update(y_true_out, indexes.stack(), updates.stack())

def read_label(file, anchors, anchor_mask, size, batch_size, ori_width, ori_height, max_size):
    # https://github.com/zzh8829/yolov3-tf2/blob/3925682b0cc7552fcef72b7b305577b8ad6c839e/yolov3_tf2/dataset.py
    y_out = []
    box_info = []
    grid_size = size // 32
    y_true = []
    for b in range(batch_size):
        line = (file[b].numpy()).decode('utf-8')
        line = line.split('[')[1]
        line = line.split(']')[0]
        line = line.split('\\n')[0]
        
        # "'177'"
        xmin = line.split(',')[0]
        xmin = xmin.split("'")[1]
        xmin = float(xmin)

        ymin = line.split(",")[1]
        ymin = ymin.split("'")[1]
        ymin = float(ymin)

        b_width = line.split(",")[2]
        b_width = b_width.split("'")[1]
        b_width = float(b_width)

        b_height = line.split(",")[3]
        b_height = b_height.split("'")[1]
        b_height = float(b_height)

        normalized_xmin = xmin / ori_width[b]
        normalized_ymin = ymin / ori_height[b]
        normalized_xmax = (xmin + b_width) / ori_width[b]
        normalized_ymax = (ymin + b_height) / ori_height[b] # 여기까지는 어떻게든 만들었음 이번에는 shape이 문제임

        #label = [int((line.split(',')[i]).split("'")[1]) for i in range(4, 44)]

        box = []
        for i in range(4, 44):
            box.append([normalized_xmin, normalized_ymin, normalized_xmax, normalized_ymax, 
                        int((line.split(',')[i]).split("'")[1])])
        box = np.array(box, dtype=np.float32)
        paddings = [[0, max_size - box.shape[0]], [0, 0]]
        box_info.append(np.pad(box, paddings, 'constant', constant_values=0))
        
    #
    box_info = np.array(box_info, dtype=np.float32)
    #
    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]
    #
    box_wh = box_info[..., 2:4] - box_info[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2), (1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    #
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * tf.minimum(box_wh[..., 1], anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)
    #
    anchor_idx = tf.cast(tf.argmax(iou, -1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, -1)
    bbox = tf.concat([box_info, anchor_idx], -1)
    #

    for anchor_idxs in anchor_mask:
        y_true.append(transform_targets_for_output(bbox, grid_size, anchor_idxs))
        grid_size *=2

    y_outs = y_true

    return tuple(y_outs)