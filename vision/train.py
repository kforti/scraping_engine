import json
import os
import numpy as np
import torch
from PIL import Image

import transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class LabelboxWebpageDataset:
    def __init__(self, root, label_path, mask_map, transforms):
        self.root = root
        self.labels = []
        self.images = []
        self.mask_map = mask_map
        self.transforms = transforms

        with open(label_path, "r") as f:
            data = json.load(f)
        for d in data:
            self.images.append(d["External ID"])
            self.labels.append(d["Label"]["objects"])

    def __getitem__(self, item):
        img_path = os.path.join(self.root, self.images[item])
        img = Image.open(img_path).convert("RGB")
        data = self.labels[item]
        boxes = []
        classes = []
        for d in data:
            classes.append(d["value"])
            bbox = d["bbox"]
            xmin = bbox["left"]
            xmax = bbox["left"] + bbox["width"]
            ymin = bbox["top"]
            ymax = bbox["top"] + bbox["height"]
            boxes.append([xmin, ymin, xmax, ymax])

        mask = self.create_mask(img, boxes, classes)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]
        # Does not include background class
        #classes = [self.mask_map[c] for c in classes]

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(obj_ids, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([item])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])


        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    def create_mask(self, img, boxes, classes):
        import numpy as np
        mask = np.zeros(img.size)
        for box, cls in zip(boxes, classes):

            value = self.mask_map[cls]
            mask[box[0]: box[2], box[1]: box[3]] = value
            #masks.append(mask)
        #masks = np.array(masks)
        return mask

import utils
from model import get_model_instance_segmentation
from engine import train_one_epoch, evaluate

def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    # our dataset has two classes only - background and person
    num_classes = 6
    # use our dataset and defined transformations
    MASK_MAP = {"title": 1,
                "rating": 2,
                "ratings": 3,
                "price": 4,
                "description": 5}

    dataset = LabelboxWebpageDataset("../data/product_images", "../data/labelbox-export.862Z.json", MASK_MAP,
                                     get_transform(True))

    dataset_test = LabelboxWebpageDataset("../data/product_images", "../data/labelbox-export.862Z.json", MASK_MAP,
                                     get_transform(False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-5])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-5:])

    # define training and validatio n data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=0.005,
    #                             momentum=0.9, weight_decay=0.0005)

    optimizer = torch.optim.Adam(params, lr=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset

        # Evaluation broken
        if epoch == num_epochs - 1:
            model.eval()
        # cpu_device = torch.device("cpu")
        # model.to(cpu_device)
        # for images, targets in data_loader_test:
        #     # images = list(img.to(cpu_device) for img in images)
        #     # outputs = model(images)
        #     # print(outputs)
        #     # print(targets)
            evaluate(model, data_loader_test, device=device)



main()
