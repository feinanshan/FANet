import os
import torch
import numpy as np
import imageio as m
import cv2
from torch.utils import data
import torchvision.transforms.functional as TF
from ptsemseg.utils import recursive_glob
from ptsemseg.augmentations import Compose, RandomHorizontallyFlip, RandomRotate, Scale

def color_map( N=40, normalized=False):
    """
    Return Color Map in PASCAL VOC format
    """

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = "float32" if normalized else "uint8"
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255.0 if normalized else cmap
    return cmap

class ade20kLoader(data.Dataset):

    colors = color_map(N=256)
    label_colours = dict(zip(range(256), colors))

    def __init__(
        self,
        root,
        split="train",
        augmentations=None,
        test_mode=False,
        model_name=None
    ):
        self.root = root
        self.split = split
        self.augmentations = augmentations
        self.test_mode=test_mode
        self.model_name=model_name
        self.n_classes = 150
        self.files = {}

        self.images_base = os.path.join(self.root, "IMG", self.split)
        self.annotations_base = os.path.join(self.root, "GT", self.split)

        self.files[split] = recursive_glob(rootdir=self.annotations_base, suffix=".png")

        self.void_classes = [0]
        self.valid_classes = range(1,151)
        self.class_names = []

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(150)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        lbl_path = self.files[self.split][index].rstrip()
        img_path = os.path.join(
            self.images_base,
            os.path.basename(lbl_path))[:-3]+'jpg'

        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)
        if len(img.shape)==2:
            img = img[:,:, np.newaxis]
            img = img.repeat(3, axis=2)

        h,w,c = img.shape
        
        lbl = m.imread(lbl_path)
        lbl = cv2.resize(lbl,(w,h), interpolation = cv2.INTER_NEAREST)
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))

        '''import cv2
        cv2.namedWindow("Image")
        cv2.imshow("Image", self.decode_segmap(lbl))
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        
        if self.test_mode:
            return img, lbl, os.path.basename(lbl_path)[:-3], h, w
        return img, lbl

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        mask_ = mask.copy()
        mask_ = mask_*0
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask_[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask_[mask == _validc] = self.class_map[_validc]
        return mask_.astype(np.uint8)

    def decode_pred(self, mask):
        # Put all void classes to zero
        for _predc in range(self.n_classes):
            mask[mask == _predc] = self.valid_classes[_predc]
        return mask.astype(np.uint8)
