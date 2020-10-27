import os
import random

import cv2

import json

import numpy as np

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode


def get_product_page_dicts(label_path, img_dir):
    MASK_MAP = {"title": 0,
                "rating": 1,
                "ratings": 2,
                "price": 3,
                "description": 4}
    images = []
    labels = []

    with open(label_path, "r") as f:
        data = json.load(f)
    for d in data:
        images.append(os.path.join(img_dir, d["External ID"]))
        try:
            l = d["Label"]["objects"]
        except:
            l = []
        labels.append(l)

    dataset_dicts = []
    for idx, (path, v) in enumerate(zip(images, labels)):
        record = {}

        height, width = cv2.imread(path).shape[:2]

        record["file_name"] = path
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width



        objs = []
        for d in v:
            bbox = d["bbox"]
            xmin = bbox["left"]
            xmax = bbox["left"] + bbox["width"]
            ymin = bbox["top"]
            ymax = bbox["top"] + bbox["height"]
            poly = [xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]

            obj = {
                "bbox": [xmin, ymin, xmax, ymax],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": MASK_MAP[d["value"]],
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


for d in ["train", "val"]:

    DatasetCatalog.register("products_" + d, lambda d=d: get_product_page_dicts("/home/kevin/bin/scraping_engine/data/export-2020-10-22T19_24_01.639Z.json",
                           "/home/kevin/bin/scraping_engine/data/product_images_2"))
    MetadataCatalog.get("products_" + d).set(thing_classes=["title", "rating", "ratings", "price", "description"])
products_metadata = MetadataCatalog.get("products_train")
dicts = get_product_page_dicts("/home/kevin/bin/scraping_engine/data/export-2020-10-22T19_24_01.639Z.json",
                           "/home/kevin/bin/scraping_engine/data/product_images_2")
for i, d in enumerate(random.sample(dicts, 3)):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=products_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    print(out)
    cv2.imwrite(f"test-img{i}.png", out.get_image()[:, :, ::-1])

    # DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
    # MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
balloon_metadata = MetadataCatalog.get("balloon_train")

from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
cfg.DATASETS.TRAIN = ("products_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 1
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()