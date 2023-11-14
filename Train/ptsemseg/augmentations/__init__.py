from ptsemseg.augmentations.augmentations import (
    AdjustContrast,
    AdjustGamma,
    AdjustBrightness,
    AdjustSaturation,
    AdjustHue,
    RandomCrop,
    RandomHorizontallyFlip,
    RandomVerticallyFlip,
    Scale,
    RandomScale,
    RandomSizedCrop,
    RandomRotate,
    RandomTranslate,
    CenterCrop,
    Compose,
    Normalize,
    Color_Normalize,
    BGR,
)


key2aug = {
    "gamma": AdjustGamma,
    "hue": AdjustHue,
    "brightness": AdjustBrightness,
    "saturation": AdjustSaturation,
    "contrast": AdjustContrast,
    "rcrop": RandomCrop,
    "hflip": RandomHorizontallyFlip,
    "vflip": RandomVerticallyFlip,
    "scale": Scale,
    "rscale": RandomScale,
    "rsizecrop": RandomSizedCrop,
    "rotate": RandomRotate,
    "translate": RandomTranslate,
    "ccrop": CenterCrop,
    "cnorm": Color_Normalize,
    "norm": Normalize,
    "bgr": BGR,
}


def get_composed_augmentations(aug_dict):


    augmentations = []
    for aug_key, aug_param in aug_dict.items():
        augmentations.append(key2aug[aug_key](aug_param))
    return Compose(augmentations)
