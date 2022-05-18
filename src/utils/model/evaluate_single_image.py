import cv2
import matplotlib

from utils.model.SatelliteTurbinesDataset import Net
matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from cv2 import waitKey
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
from PIL import Image
import torch.nn as nn
from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from torch.nn.functional import normalize
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import visualization as viz
import torchvision.transforms as transforms
from torchvision import models
def eval_image_with_model(path_to_model, imgArr, basePath, pretrained = True):
    transform=transforms.Compose([transforms.ToPILImage(),
                    transforms.Resize(255),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]
                )
    
    transformForImage = transforms.Compose([transforms.ToPILImage(),transforms.Resize((40,40)),transforms.ToTensor()])

    device = (torch.device('cuda') if torch.cuda.is_available()
        else torch.device('cpu'))
    stacked = (np.dstack((imgArr[0],imgArr[1],imgArr[2]))).astype(np.uint8)
    item=transformForImage(stacked)
    item = item[:,20:40,10:30]
    if(pretrained):
        item = transform(item)
    classes, out = run_inference(item.to(device=device),device,path_to_model, basePath=basePath)
    return classes, out

def run_inference(in_tensor, device, modelPath, basePath, pretrained = True):
    classes = ["spinning", "undetected"]

    model = models.densenet121(pretrained=True)
    model.classifier = nn.Sequential(nn.Linear(1024,512),
    nn.ReLU(),nn.Dropout(0.2),nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.1),nn.Linear(256,2))
    model.to(device=device)
    #model = Net().to(device = device) 

    #model = Net().to(device=device)
    model.load_state_dict(torch.load(modelPath))
    model.eval()
    with torch.no_grad():
        out_tensor = model(in_tensor.unsqueeze(0)).squeeze(0)
        probs = out_tensor.tolist()
        print(f'Spin: {probs[0]}, Undetected: {probs[1]}' )
        out = probs.index(max(probs))
        #we use notOut in order to reverse the heatmap and get the sureness of uncertainty
        if(pretrained==False):
            heatmap = occlusion(model,in_tensor.unsqueeze(0),out)
            mean, std, var = torch.mean(heatmap), torch.std(heatmap), torch.var(heatmap)
            heatmap = (heatmap-mean)/std
            ax = sns.heatmap(heatmap, xticklabels=False, yticklabels=False,vmax=1,vmin=0)
            figure = ax.get_figure()
            plt.figure()
            figure.savefig(basePath +"src/static/gifs/heatmap.png")
        # else: 
        #     heatmap = occlusion(model,in_tensor.unsqueeze(0),out,180,6)
        #     mean, std, var = torch.mean(heatmap), torch.std(heatmap), torch.var(heatmap)
        #     heatmap = (heatmap-mean)/std
        #     ax = sns.heatmap(heatmap, xticklabels=True, yticklabels=True,vmax=1,vmin=0)
        #     figure = ax.get_figure()
        #     plt.figure()
        #     figure.savefig(basePath +"src/static/gifs/heatmap.png")
        model.train()
        return classes[out], out

def occlusion(model, image, label, occ_size = 9, occ_stride = 1, occ_pixel = 127):
    # occ_size = 50, occ_stride = 50, occ_pixel = 0.5
    #get the width and height of the image
    width, height = image.shape[-2], image.shape[-1]
    #setting the output image width and height

    output_height = int(np.ceil((height-occ_size)/occ_stride))
    output_width = int(np.ceil((width-occ_size)/occ_stride))
    #create a white image of sizes we defined
    heatmap = torch.zeros((output_height,output_width))
    model.eval()
    #iterate all the pixels in each column
    for h in range(0, height):
        if(min(height, h*occ_stride + occ_size) >= height):
            continue
        for w in range(0, width):
            h_start = h*occ_stride
            w_start = w*occ_stride
            h_end = min(height, h_start + occ_size)
            w_end = min(width, w_start + occ_size)
            
            if (w_end) >= width or (h_end) >= height:
                continue
            
            input_image = image.clone().detach()
            
            #replacing all the pixel information in the image with occ_pixel(grey) in the specified location
            input_image[:, :, w_start:w_end, h_start:h_end] = occ_pixel
            with torch.no_grad():
                #run inference on modified image
                output = model(input_image).squeeze(0)
                probab = output.tolist()
                #wie steht der eine Output im verhältnis zum anderen?
                #setting the heatmap location to probability value
            heatmap[h,w] = probab[label] 
    model.train()
    return heatmap