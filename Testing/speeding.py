import torch
import oyaml as yaml
from torchstat import stat
import time,os

from models.bisenet.bisenet import BiSeNet
from models.swiftnet.semseg import SwiftNet
from models.icnet.icnet import ICNet 
from models.erfnet.erfnet import ERFNet
from models.segnet.segnet import SegNet
from models.shelfnet.shelfnet import ShelfNet18
from models.fanet.fanet import FANet

#os.environ["CUDA_VISIBLE_DEVICES"]="1"

def run(model,size,name):
    model.cuda()
    model.eval()
    t_cnt = 0.0
    with torch.no_grad():

        input = torch.rand(size).cuda()
        torch.cuda.synchronize()
        x = model(input)
        x = model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start_ts = time.time()

        for i in range(100):
            x = model(input)
        torch.cuda.synchronize()
        end_ts = time.time()

        t_cnt = end_ts-start_ts #t_cnt + (end_ts-start_ts)
    print("=======================================")
    print("Model Name: "+name)
    print("FPS: %f"%(100/t_cnt))
    #print("=======================================")

if __name__ == "__main__":

    segnet = SegNet()
    run(segnet,size=(1,3,360,640),name='SegNet')

    icnet = ICNet()
    run(icnet,size=(1,3,1025,2049),name='ICNet')

    erfnet = ERFNet()
    run(erfnet,size=(1,3,512,1024),name='ERFNet')

    bisenet = BiSeNet()
    run(bisenet,size=(1,3,768,1536),name='BiSeNet')
    
    shelfnet = ShelfNet18()
    run(shelfnet,size=(1,3,1024,2048),name='ShelfNet')

    swiftnet = SwiftNet()
    run(swiftnet,size=(1,3,1024,2048),name='SwiftNet')

    fanet18 = FANet(backbone='resnet18')
    run(fanet18,size=(1,3,1024,2048),name='FANet-18')

    fanet34 = FANet(backbone='resnet34')
    run(fanet34,size=(1,3,1024,2048),name='FANet-34')


    print("=======================================")
    print("Note: ")
    print("In the paper, a single Titan X GPU is adopted for evaluation.")
    print("=======================================")

