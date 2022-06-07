import cv2
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
from torchvision.utils import save_image
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.transforms as transforms
from torchvision import models
def eval_image_with_model(path_to_model, path_heatmap,path_overlay, imgArr, pretrained = True):
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
    classes, out, probs = run_inference(item.to(device=device),device,path_to_model,path_heatmap,path_overlay)
    return classes, out, probs

def run_inference(in_tensor, device, modelPath,path_heatmap,path_overlay, pretrained = True):
    classes = ["spinning", "undetected"]

    model = models.densenet121(pretrained=True)
    model.classifier = nn.Sequential(nn.Linear(1024,512),
    nn.ReLU(),nn.Dropout(0.2),nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.1),nn.Linear(256,2),nn.Sigmoid())
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
            ax = sns.heatmap(heatmap, xticklabels=False, yticklabels=False,vmax=1,vmin=0)
            figure = ax.get_figure()
            plt.figure()
            figure.savefig(path_heatmap)
        else: 
            origWidth, origHeight = in_tensor.unsqueeze(0).shape[-2], in_tensor.unsqueeze(0).shape[-1]
            #(origWidth/20) = 10m turbine = 
            occ = int((origWidth/20)*7)
            stride = 9
            heatmap, ho,wo,img = occlusion(model,in_tensor.unsqueeze(0),out,occ,stride)
            print(ho,wo)
            heatmap = heatmap.numpy()
            #width = [round((element/wiNo), 1) for element in width]
            x = []
            my_xticks = []
            for i in range(0,8):
                x.append(28*i)
                my_xticks.append(2.5*i)
            #height = [round((element/wiNo), 1) for element in height]
            ax = plt.imshow(heatmap, cmap='hot', interpolation='nearest')
            plt.xticks(x, my_xticks)
            plt.yticks(x, my_xticks)
            figure = ax.get_figure()
            plt.figure(figsize=(20, 20))
            figure.savefig(path_heatmap)

            img = torch.squeeze(img.to(device="cpu"))
            v_min, v_max = img.min(), img.max()
            new_min, new_max = 0, 1
            img = (img - v_min)/(v_max - v_min)*(new_max - new_min) + new_min
            imgnp = list(np.array(img))
            ax = plt.imshow(cv2.merge([imgnp[2],imgnp[1],imgnp[0]]))
            plt.xticks(x, my_xticks)
            plt.yticks(x, my_xticks)
            figure = ax.get_figure()
            plt.figure(figsize=(20, 20))
            figure.savefig(path_overlay)
            
            
        model.train()
        return classes[out], out, probs

def occlusion(model, image, label, occ_size = 9, occ_stride = 1, occ_pixel = 0):
    # occ_size = 50, occ_stride = 50, occ_pixel = 0.5
    #get the width and height of the image
    width, height = image.shape[-2], image.shape[-1]
    #setting the output image width and height

    output_height = int(np.ceil((height-occ_size)/occ_stride))
    output_width = int(np.ceil((width-occ_size)/occ_stride))
    #create a white image of sizes we defined
    print(height, width)
    #print(output_height,output_width)

    heatmap = torch.full((height,width),-1.0)
    model.eval()
    minprob = 1
    img = image.clone().detach()
    #iterate all the pixels in each column
    # saveRow = []
    # saveCol = []
    for h in range(0, height):
        if(min(height, h*occ_stride + occ_size) >= height):
            #saveRow.append(h)
            continue
        for w in range(0, width):
            h_start = h*occ_stride
            w_start = w*occ_stride
            h_end = min(height, h_start + occ_size)
            w_end = min(width, w_start + occ_size)
            
            if (w_end) >= width:
                #saveCol.append(w)
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
                if(probab[label]<minprob):
                    minprob = probab[label]
                    img = input_image.clone().detach()
                heatmap[((h_start+h_end)//2-(occ_stride//2)):((h_start+h_end)//2+(occ_stride//2)),(w_start+w_end)//2-(occ_stride//2):(w_start+w_end)//2+(occ_stride//2)] += probab[label]
                # noLabel = 0
                # if(label == 0):
                #     noLabel = 1
                # if(probab[noLabel]>probab[label]):
                #     heatmap[h,w] = probab[label] 
                # else:
                #     heatmap[h,w] = 1
    model.train()
    return heatmap,output_height,output_width,img