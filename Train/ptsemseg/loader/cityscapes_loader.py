import os
import torch
import numpy as np
import imageio as m
import pdb

from torch.utils import data
import torchvision.transforms.functional as TF
from ptsemseg.utils import recursive_glob
from ptsemseg.augmentations import Compose, RandomHorizontallyFlip, RandomRotate, Scale


class cityscapesLoader(data.Dataset):
    """cityscapesLoader

    https://www.cityscapes-dataset.com

    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/

    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """

    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colours = dict(zip(range(19), colors))



    def __init__(
        self,
        root,
        split,
        augmentations=None,
        model_name=None,
        test_mode = False
    ):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.test_mode = test_mode
        self.root = root
        self.split = split
        self.augmentations = augmentations
        self.model_name=model_name
        self.n_classes = 19
        self.files = self.read_list()


        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))

        if len(self.files)==0:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d images in: %s" % (len(self.files),split))

    def read_list(self):
        list_ = []
        for line in open(self.split): 
            img, ann =  line.strip('\n').split('\t')
            list_.append([os.path.join(self.root,img),os.path.join(self.root,ann)])
        return list_


    def __len__(self):
        """__len__"""
        return len(self.files)

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[index][0].rstrip()
        lbl_path = self.files[index][1].rstrip()

        lbl_name = lbl_path.split('/')[-1]

        
        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        h_,w_,c_ = img.shape

        lbl = m.imread(lbl_path)
        if len(lbl.shape)==3:
            lbl = lbl[:,:,0]
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        '''import cv2
        cv2.namedWindow("Image")
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        pdb.set_trace()'''
    
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        if self.test_mode:
            return img, lbl, lbl_name, w_, h_
        else:
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
        rgb[:, :, 0] = b / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = r / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def decode_pred(self, mask):
        # Put all void classes to zero
        mask_ = mask.copy()
        mask_ = mask_*0
        for _predc in range(self.n_classes):
            mask_[mask == _predc] = self.valid_classes[_predc]
        return mask_.astype(np.uint8)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    augmentations = Compose([Scale(2048), RandomRotate(10), RandomHorizontallyFlip(0.5)])

    local_path = "/datasets01/cityscapes/112817/"
    dst = cityscapesLoader(local_path, is_transform=True, augmentations=augmentations)
    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
    for i, data_samples in enumerate(trainloader):
        imgs, labels = data_samples
        import pdb

        pdb.set_trace()
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0, 2, 3, 1])
        f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
        plt.show()
        a = input()
        if a == "ex":
            break
        else:
            plt.close()
