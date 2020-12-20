# _*_coding : UTF-8_*_
# 开发人员 bruno.zhao
# 开发时间 2020/12/10  下午6:12
# 文件名称 prediction.py
# 开发工具 PyCharm
import time
import utils
import numpy as np
import torch
from PIL import Image, ImageDraw  # python  图像处理模块
from engine import predict, train_one_epoch
from train_frcnn import PennFudanDataset, get_transform, device, dataset_test, dataset
import random


# To calculate the iou
def calculate_iou(box1, box2):

    box1 = np.array(box1.cpu())
    box2 = np.array(box2.cpu())

    box1_x1, box1_y1, box1_x2, box1_y2 = box1[0], box1[1], box1[2], box1[3]
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[0], box2[1], box2[2], box2[3]

    cross_x1 = max(box1_x1, box2_x1)
    cross_y1 = max(box1_y1, box2_y1)
    cross_x2 = min(box1_x2, box2_x2)
    cross_y2 = min(box1_y2, box2_y2)

    cross_area = max(cross_x2 - cross_x1 + 1, 0) * max(cross_y2 - cross_y1 + 1, 0)
    # print('cross_area', cross_area)
    box1_area = (box1_x2 - box1_x1 + 1) * (box1_y2 - box1_y1 + 1)
    box2_area = (box2_x2 - box2_x1 + 1) * (box2_y2 - box2_y1 + 1)

    iou = cross_area / (box1_area + box2_area - cross_area)

    return iou


# for add_box(image, index, color)
def add_boxes(im, boxes, str1, str2):
    draw = ImageDraw.Draw(im)
    for box in boxes:
        box = np.array(box.cpu())
        draw.rectangle(box, None, str1, width=2)
        draw.text(box[:2], str2, str1)
        pass
    pass


def add_box(draw, box, str1, str2):
    box = np.array(box.cpu())
    draw.rectangle(box, None, str1, width=2)
    draw.text(box[:2], str2, str1)
    pass


def compare_result(img_p, box1, box2, AP_value):
    draw = ImageDraw.Draw(img_p)
    for box_1 in box1:
        str1 = 'red'
        str2 = '  wrong'
        for box_2 in box2:
            # print(calculate_iou(box_1, box_2))
            if calculate_iou(box_1, box_2) > AP_value:
                str1 = 'green'
                str2 = '  IOU = '+str(format(calculate_iou(box_1, box_2), '.4f'))
                pass
            pass
        add_box(draw, box_1, str1, str2)
        pass
    pass


def get_mask(ptest, img_p, threshold):
    mask = ptest[0]['masks']
    mask = np.array(mask.cpu()).squeeze(1)
    for i in range(np.shape(mask)[0]):
        test_mask = mask[i, :, :]
        test_mask[test_mask < threshold] = 0
        [change_x, change_y] = np.nonzero(test_mask)
        for j in range(len(change_x)):
            random.seed(i)
            img_p[i % 3, change_x[j], change_y[j]] = 255-round(255 * random.random())
            # img_p[1, change_x[j], change_y[j]] = 255 - round(255 * random.random())
            pass
        pass
    print('----------------', mask.shape)

    pass


def get_mask_ground(data, img_p, threshold):
    mask = data[1]['masks']
    mask = np.array(mask.cpu())
    for i in range(np.shape(mask)[0]):
        test_mask = mask[i, :, :]
        test_mask[test_mask < threshold] = 0
        [change_x, change_y] = np.nonzero(test_mask)
        for j in range(len(change_x)):
            random.seed(i)
            img_p[i % 3, change_x[j], change_y[j]] = 255-round(255 * random.random())
            # img_p[1, change_x[j], change_y[j]] = 255 - round(255 * random.random())
            pass
        pass
    print('----------------', mask.shape)

    pass


model_path = './model.pkl'
model = torch.load(model_path)
count = 0
for data in dataset_test:
    ptest = predict(data, device, model_path)

    # predict box
    boxes_p = ptest[0]['boxes']
    # ground truth box
    boxes_g = data[1]['boxes']

    # for the original img
    img_p, _ = data
    get_mask(ptest, img_p, 0.35)
    # get_mask_ground(data, img_p, 0.5)

    img_p = img_p.permute(1, 2, 0)
    img_p = Image.fromarray(img_p.mul(255).byte().cpu().numpy())
    # add_boxes(img_p, boxes_g, 'black', 'Ground_truth')
    # add_boxes(img_p, boxes_p, 'green', 'prediction')

    # for result
    compare_result(img_p, boxes_p, boxes_g, 0.75)
    # img_p.save('./ground_truth/00' + str(count) + 'out.png')
    img_p.save('./prediction_output/00' + str(count) + 'out.png')
    count = count + 1
    pass


