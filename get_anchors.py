# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import easydict
import os

# https://github.com/lars76/kmeans-anchor-boxes/blob/master/example.py

FLAGS = easydict.EasyDict({"img_size": 416,
                           
                           "tr_img_path": "D:/[1]DB/[3]detection_DB/Celeb/archive/img_celeba/img_celeba.7z/img_celeba",
                           
                           "tr_txt_path": "D:/[1]DB/[3]detection_DB/Celeb/archive/label/train.txt",})

def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    :param box: tuple or array, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    :param boxes: numpy array of shape (r, 4)
    :return: numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters

def read_label(file, ori_width, ori_height):

    for b in range(1):
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

        box.append([normalized_xmax - normalized_xmin, normalized_ymax - normalized_ymin])

    return box


def input_func(img_path, lab_path):

    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, 3)
    original = img
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size]) / 255.

    return img, lab_path, (tf.shape(original)[1], tf.shape(original)[0])


tr_img = np.loadtxt(FLAGS.tr_txt_path, dtype="<U200", skiprows=0, usecols=0)
tr_img = [FLAGS.tr_img_path + "/" + data for data in tr_img]
tr_txt = open(FLAGS.tr_txt_path, "r")
tr_lab = []
for i in range(len(tr_img)):
    line = tr_txt.readline()
    line = line.split(' ')[1:]
    line = str(line)
    tr_lab.append(line)
tr_lab = np.array(tr_lab)
print(len(tr_img), len(tr_lab))

tr_generator = tf.data.Dataset.from_tensor_slices((tr_img, tr_lab))
tr_generator = tr_generator.shuffle(len(tr_img))
tr_generator = tr_generator.map(input_func)
tr_generator = tr_generator.batch(1)
tr_generator = tr_generator.prefetch(tf.data.experimental.AUTOTUNE)

tr_iter = iter(tr_generator)
tr_idx = len(tr_img) // 1

box_info = []
for step in range(tr_idx):
    box = []
    batch_images, batch_labels, original_image = next(tr_iter)
    width = original_image[0]
    height = original_image[1]
    width = np.array(width, dtype=np.float32)
    height = np.array(height, dtype=np.float32)

    line = (batch_labels[0].numpy()).decode('utf-8')
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

    normalized_xmin = xmin / width[0]
    normalized_ymin = ymin / height[0]
    normalized_xmax = (xmin + b_width) / width[0]
    normalized_ymax = (ymin + b_height) / height[0]

    box.append([normalized_xmax - normalized_xmin, normalized_ymax - normalized_ymin])

    for i in range(len(box)):
        box_info.append(box[i])

    if step % 10000 == 0:
        print(step)

box_info = np.array(box_info, dtype=np.float32)
out = kmeans(box_info, k=9)
print(out)
print("Accuracy: {:.2f}%".format(avg_iou(box_info, out) * 100))
print("Boxes:\n {}".format(out))

ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
print("Ratios:\n {}".format(sorted(ratios)))