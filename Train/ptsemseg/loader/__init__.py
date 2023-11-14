import json
from ptsemseg.loader.cityscapes_loader import cityscapesLoader
from ptsemseg.loader.context_loader import contextLoader
from ptsemseg.loader.ade20k_loader import ade20kLoader
from ptsemseg.loader.cocostuff_loader import cocostuffLoader
from ptsemseg.loader.ccri_loader import CCRILoader


def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        "cityscapes": cityscapesLoader,
        "pascal_context": contextLoader,
        "ade20k": ade20kLoader,
        "cocostuff": cocostuffLoader,
        "ccri": CCRILoader,
    }[name]
