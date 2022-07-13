import cv2
import matplotlib
from zmq import device

from utils.model.SatelliteTurbinesDataset import Net
matplotlib.use('Agg')
import numpy as np
import torch
from torchvision.utils import save_image,make_grid
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as f
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary


def eval_image_with_model(path_to_model, path_heatmap,path_overlay, imgArr, pretrained = True):

    # transform=transforms.Compose([transforms.ToPILImage(),
    #                 transforms.Resize(255),
    #                 transforms.CenterCrop(224),
    #                 transforms.ToTensor(),
    #                 transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]
    #             )
    transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((40,40)),transforms.ToTensor()])

    
    transformForImage = transforms.Compose([transforms.ToPILImage(),transforms.Resize((40,40)),transforms.ToTensor()])

    device = (torch.device('cuda') if torch.cuda.is_available()
        else torch.device('cpu'))
    stacked = (np.dstack((imgArr[0],imgArr[1],imgArr[2]))).astype(np.uint8)
    item=transformForImage(stacked)
    print("here1")
    item = item[:,20:40,10:30]
    if(pretrained):
        item = transform(item)
    classes, out, probs = run_inference(item.to(device=device),device,path_to_model,path_heatmap,path_overlay)
    return classes, out, probs

def run_inference(in_tensor, device, modelPath,path_heatmap,path_overlay, pretrained = False):
    classes = ["spinning", "undetected"]

    # model = models.densenet121(pretrained=True)
    # model.classifier = nn.Sequential(nn.Linear(1024,512),
    # nn.ReLU(),nn.Dropout(0.2),nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.1),nn.Linear(256,2),nn.Sigmoid())
    # model.to(device=device)
    model = Net().to(device = device) 
    model.load_state_dict(torch.load(modelPath))
    model.eval()
    with torch.no_grad():
        print(model(in_tensor.unsqueeze(0)))
        out_tensor = f.softmax(model(in_tensor.unsqueeze(0)).squeeze(0), dim=0)
        probs = out_tensor.tolist()
        print(f'Spin: {probs[0]}, Undetected: {probs[1]}' )
        out = probs.index(max(probs))

        

        print("here")
        #we use notOut in order to reverse the heatmap and get the sureness of uncertainty
        if(pretrained==True):
            heatmap = occlusion(model,in_tensor.unsqueeze(0),out)
            ax = sns.heatmap(heatmap, xticklabels=False, yticklabels=False,vmax=1,vmin=0)
            figure = ax.get_figure()
            plt.figure()
            figure.savefig(path_heatmap)
        else: 
            origWidth, origHeight = in_tensor.unsqueeze(0).shape[-2], in_tensor.unsqueeze(0).shape[-1]
            #(origWidth/20) = 10m turbine = 
            occ = 10
            stride = 1
            heatmap, ho,wo,img = occlusion(model,in_tensor.unsqueeze(0), out,occ,stride)
            print(ho,wo)
            v_min, v_max = heatmap.min(), heatmap.max()
            new_min, new_max = 0, 1
            heatmap = torch.abs(1-((heatmap - v_min)/(v_max - v_min)*(new_max - new_min) + new_min))
            heatmap = heatmap.numpy()
            #width = [round((element/wiNo), 1) for element in width]
            # x = []
            # my_xticks = []
            # for i in range(0,8):
            #     x.append(28*i)
            #     my_xticks.append(2.5*i)
            #height = [round((element/wiNo), 1) for element in height]
            ax = plt.imshow(heatmap, cmap='hot',interpolation='nearest')
            cb = plt.colorbar()
            cb.set_label('Oclusion sensitivity for a 10 pixel occlusion')

            # Add title and labels to plot.
            figure = ax.get_figure()
            plt.figure()
            figure.savefig(path_heatmap)
            print("heatmap Done")

            img = torch.squeeze(img.to(device="cpu"))
            v_min, v_max = img.min(), img.max()
            new_min, new_max = 0, 1
            img = (img - v_min)/(v_max - v_min)*(new_max - new_min) + new_min
            imgnp = list(np.array(img))
            ax = plt.imshow(cv2.merge([imgnp[2],imgnp[1],imgnp[0]]))
            figure = ax.get_figure()
            plt.figure(figsize=(20, 20))
            figure.savefig(path_overlay)
            print("here2")

        # layer = 0
        # filter = model.features[layer].weight.data.clone().detach().cpu()
        # visTensor(filter, ch=0, allkernels=False)
  
        model.train()
        return classes[out], out, probs

def occlusion(model, image, label, occlusion = 10, offset = 1, color = 0.5):
    #get the width and height of the image
    width, height = image.shape[-2], image.shape[-1]
  
    #setting the output image width and height
    output_height = int(np.ceil((height-occlusion)/offset))
    output_width = int(np.ceil((width-occlusion)/offset))
  
    #create a white image of sizes we defined
    heatmap = torch.zeros((height, width))
    model.eval()
    minprob = 1
    img = image.clone().detach()
    #iterate all the pixels in each column
    # saveRow = []
    # saveCol = []
    for h in range(0, height):
        for w in range(0, width):
            
            h_start = h*offset
            w_start = w*offset
            h_end = min(height, h_start + occlusion)
            w_end = min(width, w_start + occlusion)
            input_image = image.clone().detach()
            input_image[:, :, w_start:w_end, h_start:h_end] = color


            if (w_end) >= width and (h_end) >= height:
                input_image[:, :, w_start:width, h_start:height] = color
            elif (w_end) >= width:
                input_image[:, :, w_start:width, h_start:h_end] = color
            elif (h_end) >= height:
                input_image[:, :, w_start:w_end, h_start:height] = color

            
            
            #replacing all the pixel information in the image with occ_pixel(grey) in the specified location
            
            with torch.no_grad():
                #run inference on modified image
                output = f.softmax(model(input_image),dim=1)
                probab = output.tolist()[0]

                #wie steht der eine Output im verhältnis zum anderen?
                #setting the heatmap location to probability value
                if(probab[label]<minprob):
                    print(probab[label])
                    minprob = probab[label]
                    img = input_image.clone().detach()
                heatmap[w,h] = probab[label] 
                # noLabel = 0
                # if(label == 0):
                #     noLabel = 1
                # if(probab[noLabel]>probab[label]):
                #     heatmap[h,w] = probab[label] 
                # else:
                #     heatmap[h,w] = 1
    model.train()
    return heatmap,output_height,output_width,img

def visTensor(tensor, ch=0, allkernels=False, nrow=3, padding=1): 
        print(tensor.shape)
        n,c,w,h = tensor.shape

        if allkernels: tensor = tensor.view(n*c, -1, h,w)
        elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

        rows = np.min((tensor.shape[0] // nrow + 1, 64))    
        grid1 = make_grid(tensor, nrow=4, normalize=True, padding=padding)
        # grid2 = make_grid(tensor[:,:,1,:,:], nrow=4, normalize=True, padding=padding)
        # grid3 = make_grid(tensor[:,:,2,:,:], nrow=4, normalize=True, padding=padding)

        
        plt.imshow(grid1.numpy().transpose((1, 2, 0)))
        plt.axis('off')
        plt.ioff()
        plt.show()
        plt.savefig("Data/data_science/PLOTS/conv")  

        # plt.imshow(grid2.numpy().transpose((1, 2, 0)))
        # plt.axis('off')
        # plt.ioff()
        # plt.show()
        # plt.savefig("Data/data_science/PLOTS/conv1")  
        # plt.imshow(grid3.numpy().transpose((1, 2, 0)))
        # plt.axis('off')
        # plt.ioff()
        # plt.show()
        # plt.savefig("Data/data_science/PLOTS/conv2")  



       