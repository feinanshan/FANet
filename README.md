# FANet
Real-time Semantic Segmentation with Fast Attention

[Ping Hu](http://cs-people.bu.edu/pinghu/), [Federico Perazzi](https://fperazzi.github.io/), [Fabian Caba Heilbron](http://fabiancaba.com/), [Oliver Wang](http://www.oliverwang.info/), [Zhe Lin](http://sites.google.com/site/zhelin625/), [Kate Saenko](http://ai.bu.edu/ksaenko.html/) , [Stan Sclaroff](http://www.cs.bu.edu/~sclaroff/)

[[Paper Link](https://arxiv.org/abs/2007.03815)] [[Project Page](https://cs-people.bu.edu/pinghu/FANet.html)]

Accurate semantic segmentation requires rich contextual cues (large receptive fields) and fine spatial details (high resolution), both of which incur high computational costs. In this paper, we propose a novel architecture that addresses both challenges and achieves state-of-the-art performance for semantic segmentation of high-resolution images and videos in real-time. The proposed architecture relies on our fast attention, which is a simple modification of the popular self-attention mechanism and captures the same rich contextual information at a small fraction of the computational cost, by changing the order of operations. Moreover, to efficiently process high-resolution input, we apply an additional spatial reduction to intermediate feature stages of the network with minimal loss in accuracy thanks to the use of the fast attention module to fuse features. We validate our method with a series of experiments, and show that results on multiple datasets demonstrate superior performance with better accuracy and speed compared to existing approaches for real-time semantic segmentation. On Cityscapes, our network achieves 74.4% mIoU at 72 FPS and 75.5% mIoU at 58 FPS on a single Titan X GPU, which is ~50% faster than the state-of-the-art while retaining the same accuracy.


