# -*- coding:utf-8 -*-
from copy import deepcopy
import numpy as np
import cv2

def draw_outputs(img, outputs, class_names):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    canvas = np.zeros((1024, 1024, 3), dtype="uint8")
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (0, 0, 255), 2)
        text = '{} {:.4f}'.format(class_names[int(classes[i])], objectness[i])
        cv2.putText(canvas, text, (10, (i * 13) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


    return img, canvas

def get_model_scores(img, labels, outputs, class_names):
    # https://towardsdatascience.com/evaluating-performance-of-an-object-detection-model-137a349c517b
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]

    n_pred_box = nums
    wh = np.flip(img.shape[0:2])
    arg_sort = np.argsort(objectness)
    x1y1 = tuple((np.array(boxes[...,0:2]) * wh).astype(np.int32))
    x2y2 = tuple((np.array(boxes[...,2:4]) * wh).astype(np.int32))
    pred_boxes = np.concatenate(x1y1, x2y2, -1)
    pred_boxes = pred_boxes[arg_sort]

    pred_boxes_pruned = deepcopy(pred_boxes)

    precisions = []
    recalls = []

    gt_box = labels[0:4]
    box_score = pred_boxes_pruned

    for i in range(n_pred_box):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        
        
    

