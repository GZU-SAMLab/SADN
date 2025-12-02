#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import json
from sqlite3 import dbapi2
import numpy as np
import os
from collections import defaultdict
import cv2
import tqdm
import sys
from fvcore.common.file_io import PathManager
from PIL import Image
import torch 
current_script_path = os.path.abspath(__file__)#计算项目根目录路径(假设 tools 目录在项目根目录下，所以根目录是 tools 的上级)
project_root = os.path.dirname(os.path.dirname(current_script_path))#将项目根目录添加到 Python 的搜索路径中(强制去重，避免重复添加)
if project_root not in sys.path:
    sys.path.insert(0,project_root)
from detectron2.data import DatasetCatalog, MetadataCatalog
from dcfs.data import register_all_coco
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer


class Visualizer(Visualizer):
    def __init__(self, img_rgb, metadata=None, scale=1.0, instance_mode=None):
        # 调用父类初始化
        super().__init__(img_rgb, metadata=metadata, scale=scale, instance_mode=instance_mode)
        # 自定义可视化参数
        self._default_font_size = 24  # 增大字体
        self._default_line_width = 10  # 加粗边界框
       


    def create_text_labels(self, classes, scores, class_names, is_crowd=None):
        """
        自定义标签生成函数，只显示类别名称，不显示置信度分数
        """
        labels = None
        if classes is not None:
            if class_names is not None and len(class_names) > 0:
                labels = [class_names[i] for i in classes]
            else:
                labels = [str(i) for i in classes]

        # 不再添加 scores
        if labels is not None and is_crowd is not None:
            labels = [l + ("|crowd" if crowd else "") for l, crowd in zip(labels, is_crowd)]
        return labels

    def draw_dataset_dict(self, dic):
        """
        Draw annotations/segmentaions in Detectron2 Dataset format.
        """
        annos = dic.get("annotations", None)
        if annos:
            boxes = [
                BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYXY_ABS)
                if len(x["bbox"]) == 4
                else x["bbox"]
                for x in annos
            ]
        
            category_ids = [x["category_id"] for x in annos]
            colors = []
            for x in annos:
                colors.append([0, 0, 1])  # 蓝色表示真实标注

            if hasattr(self, 'metadata') and hasattr(self.metadata, 'thing_classes'):
                class_names = self.metadata.thing_classes
            else:
                class_names = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]
            
            labels = self.create_text_labels(
                category_ids,
                scores=None,
                class_names=class_names,
                is_crowd=None,
            )

            # 修改这里：增加线条粗细和字体大小
            self.overlay_instances(
                labels=labels, 
                boxes=boxes, 
                masks=None,
                keypoints=None,
                assigned_colors=colors,
                alpha=0.8
            )

            # 手动设置线条与字体大小（Detectron2 默认不暴露）
            for ann in self.output.ax.patches:
                ann.set_linewidth(self._default_line_width)
            for text in self.output.ax.texts:
                text.set_fontsize(self._default_font_size)

        return self.output

def create_instances(predictions, image_size):
    ret = Instances(image_size)
    score = np.asarray([x["score"] for x in predictions])
    chosen = (score > args.conf_threshold).nonzero()[0]
    if chosen.size == 0:
        # 如果没有满足条件的预测框，直接返回一个空的 Instances 对象
        return ret

    score = score[chosen]
    bbox = np.asarray([predictions[i]["bbox"] for i in chosen])
        
    bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

    labels = np.asarray([dataset_id_map(predictions[i]["category_id"]) for i in chosen])

    ret.scores = score
    ret.pred_boxes = Boxes(bbox)
    ret.pred_classes = labels

    # 移除预测掩码
    # 不设置 pred_masks 属性

    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that visualizes the json predictions from COCO or LVIS dataset."
    )
    parser.add_argument("--input", required=True, help="JSON file produced by the model")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--dataset", help="name of the dataset", default="coco_2017_val")
    parser.add_argument("--conf-threshold", default=0.6, type=float, help="confidence threshold")
    args = parser.parse_args()

    logger = setup_logger()

    with PathManager.open(args.input, "r") as f:
        predictions = json.load(f)

    pred_by_image = defaultdict(list)
    for p in predictions:
        pred_by_image[p["image_id"]].append(p)
    
    dicts = list(DatasetCatalog.get(args.dataset))
    
    metadata = MetadataCatalog.get(args.dataset)
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):

        def dataset_id_map(ds_id):
            return metadata.thing_dataset_id_to_contiguous_id[ds_id]

    elif "lvis" in args.dataset:
        # LVIS results are in the same format as COCO results, but have a different
        # mapping from dataset category id to contiguous category id in [0, #categories - 1]
        def dataset_id_map(ds_id):
            return ds_id - 1

    else:
        raise ValueError("Unsupported dataset: {}".format(args.dataset))
    
    os.makedirs(args.output, exist_ok=True)

    for dic in tqdm.tqdm(dicts):
    
        data_path = os.path.dirname(dic["file_name"])
        image_name = dic["file_name"].split('/')[-1]
        file_name = os.path.join(data_path,  image_name.split('_')[1], image_name)
    
        img = cv2.imread(file_name, cv2.IMREAD_COLOR)
        
        
        predictions = create_instances(pred_by_image[dic["image_id"]], img.shape[:2])
        
        # 修改这里：为真实标注也传入metadata
        vis_gt = Visualizer(img, metadata)  # 传入metadata
        vis_gt = vis_gt.draw_dataset_dict(dic).get_image()
    
        vis_pred = Visualizer(img, metadata)
        vis_pred = vis_pred.draw_instance_predictions(predictions).get_image()
    
        concat = np.concatenate((vis_pred, vis_gt), axis=1)
        cv2.imwrite(os.path.join(args.output, image_name), concat[:, :, ::-1])