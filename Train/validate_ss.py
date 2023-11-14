import os
import torch
import argparse
import timeit
import numpy as np
import oyaml as yaml
from torch.utils import data

import torch.nn.functional as F
import cv2
import pdb
from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.metrics import runningScore
from ptsemseg.utils import convert_state_dict
from ptsemseg.augmentations import get_composed_augmentations

print(torch.__version__)
torch.backends.cudnn.benchmark = False

def flip_tensor(x, dim):
    """
    Flip Tensor along a dimension
    """
    dim = x.dim() + dim if dim < 0 else dim
    return x[tuple(slice(None, None) if i != dim
                   else torch.arange(x.size(i) - 1, -1, -1).long()
                   for i in range(x.dim()))]


def upsample_predictions(pred, input_shape, scale):
    # Override upsample method to correctly handle `offset`
    result = OrderedDict()
    for key in pred.keys():
        out = F.interpolate(pred[key], size=input_shape, mode='bilinear', align_corners=True)
        if 'offset' in key:  # The order of second dim is (offset_y, offset_x)
            out *= 1.0 / scale
        result[key] = out
    return result


def multi_scale_inference(model, raw_image, device, scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2], flip=True):
#def multi_scale_inference(model, raw_image, device, scales=[1], flip=False):

    output_stride = 2 ** (5 - 0)
    scale = 1. / output_stride
    if flip:
        flip_range = 2
    else:
        flip_range = 1
    n,c, h, w = raw_image.shape
    org_h_pad = h
    org_w_pad = w

    sum_semantic_with_flip = 0
    sum_center_with_flip = 0
    sum_offset_with_flip = 0
    for i in range(len(scales)):
        image = raw_image
        scale = scales[i]
        raw_h = int(h * scale)
        raw_w = int(w * scale)
        image = F.interpolate(image, size=(raw_h, raw_w), mode='bilinear', align_corners=False)
        
        # pad image
        new_h = (raw_h) // 1 * 1
        new_w = (raw_w) // 1 * 1 
        input_image = torch.zeros(n,c,new_h, new_w).float()
        input_image[:,:, :raw_h, :raw_w] = image

        image = image.to(device)

        model = model.to(device)

        for flip in range(flip_range):
            if flip:
                image = flip_tensor(image, 3)
            out = model(image)
            semantic_pred = out[:, :, : raw_h, : raw_w]

            if raw_h != org_h_pad or raw_w != org_w_pad:
                semantic_pred = F.interpolate(semantic_pred, size=(org_h_pad, org_w_pad), mode='bilinear', align_corners=False)
   
            # average softmax or logit?
            semantic_pred = F.softmax(semantic_pred, dim=1)

  
            if flip:
                semantic_pred = flip_tensor(semantic_pred, 3)
     

            sum_semantic_with_flip += semantic_pred
   

    semantic_mean = sum_semantic_with_flip / (flip_range * len(scales))


    return semantic_mean

def validate(cfg, args):

    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]


    val_augmentations = cfg["validating"].get("val_augmentations", None)
    v_data_aug = get_composed_augmentations(val_augmentations)

    v_loader = data_loader(data_path,split=cfg["data"]["val_split"],augmentations=v_data_aug,test_mode=True)

    n_classes = v_loader.n_classes
    valloader = data.DataLoader(
        v_loader, batch_size=cfg["validating"]["batch_size"], num_workers=cfg["validating"]["n_workers"]
    )

    running_metrics = runningScore(n_classes)

    # Setup Model

    model = get_model(cfg["model"], n_classes).to(device)
    state = torch.load(cfg["validating"]["resume"])#["model_state"]
    #state = convert_state_dict(state)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    elapsed_time = 0.0

    with torch.no_grad():
        for i, (images, labels, name, w_, h_) in enumerate(valloader):

            images = images.to(device)

            torch.cuda.synchronize()
            start_ts = timeit.default_timer()

            #outputs = multi_scale_inference(model, images, device)
            outputs = model(images)
            
            torch.cuda.synchronize()
            end_ts = timeit.default_timer()
            pred = outputs.data.max(1)[1].cpu().numpy()

            if args.measure_time:
                elapsed_time_ = end_ts - start_ts

                if i > 10:
                    elapsed_time = elapsed_time+elapsed_time_
                print(
                    "Inference time \
                      (iter {0:5d}): {1:3.5f} fps".format(
                        i + 1, pred.shape[0] / elapsed_time_
                    )
                )
            if False:
                decoded = v_loader.decode_segmap(pred[0])

                cv2.namedWindow("Image")
                cv2.imshow("Image", decoded)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            gt = labels.numpy()
            running_metrics.update(gt, pred)

            '''decoded = v_loader.decode_pred(pred[0])
            decoded = cv2.resize(decoded,(w_, h_),interpolation = cv2.INTER_NEAREST)
            
            folder_ = name[0].split('_')[0]
            if not os.path.exists(cfg["validating"]["outpath"]+'/'+folder_):
                os.mkdir(cfg["validating"]["outpath"]+'/'+folder_)

            cv2.imwrite(cfg["validating"]["outpath"]+'/'+folder_+'/'+name[0], decoded)'''

    print(
        "Ave Inference time \
          {0:3.5f} fps".format(
            490 / elapsed_time
        ))
    score, class_iou = running_metrics.get_scores()

    for k, v in score.items():
        print(k, v)

    for i in range(n_classes):
        print(i, class_iou[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/fcn8s_pascal.yml",
        help="Config file to be used",
    )
    parser.add_argument(
        "--gpu",
        nargs="?",
        type=str,
        default="0",
        help="GPU ID",
    )
    parser.add_argument(
        "--eval_flip",
        dest="eval_flip",
        action="store_true",
        help="Enable evaluation with flipped image |\
                              True by default",
    )
    parser.add_argument(
        "--no-eval_flip",
        dest="eval_flip",
        action="store_false",
        help="Disable evaluation with flipped image |\
                              True by default",
    )


    parser.add_argument(
        "--measure_time",
        dest="measure_time",
        action="store_true",
        help="Enable evaluation with time (fps) measurement |\
                              True by default",
    )
    parser.add_argument(
        "--no-measure_time",
        dest="measure_time",
        action="store_false",
        help="Disable evaluation with time (fps) measurement |\
                              True by default",
    )
    parser.set_defaults(measure_time=True)

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)

    validate(cfg, args)
