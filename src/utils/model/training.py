import csv
from datetime import datetime
import torch
import torch.optim as optim

def training_loop(n_epochs, optimizer:optim.Optimizer, model, loss_fn, train_loader, device,name, path_save,val=None): 
    dateNow = datetime.now().strftime("%d%m%y-%H%M%S")
    bestEpoch, bestLoss, bestAccuracy = 0,0,0
    for epoch in range(0, n_epochs + 1):
        loss_train = 0.0
        correct=0
        total=0
        for imgs, labels in train_loader:
            imgs = imgs.to(device = device)
            labels = labels.to(device = device)
            #imgs = torch.permute(imgs,())
            outputs = model(imgs)
            loss = loss_fn(outputs, labels) #.view(-1, 1).to(torch.float32) for other loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            total += labels.shape[0]                 
            correct += int((predicted == labels).sum())

        mean_val_accuracy,mean_val_loss=onlyVal(model,val,device,loss_fn)

        if(epoch == 0):
            bestLoss = mean_val_loss
            bestAccuracy = mean_val_accuracy
        elif(loss_epoch*accu_epoch<bestLoss*bestAccuracy):
            bestLoss = loss_epoch
            bestAccuracy = accu_epoch
            bestEpoch = epoch
            
            
        if epoch % 4 == 0:
            # print('{} Epoch {}, Training loss {}'.format(
            # datetime.now(), epoch,
            # loss_train / len(train_loader)))
            loss_epoch=100.*loss_train/len(train_loader)
            accu_epoch=100.*correct/total
            print('Epoch: %.3f Train Loss: %.3f | Accuracy: %.3f'%(epoch,loss_epoch,accu_epoch))
            mean_val_accuracy,mean_val_loss=onlyVal(model,val,device,loss_fn)
            print('Epoch: %.3f Val Loss: %.3f | Accuracy: %.3f'%(epoch,mean_val_loss,mean_val_accuracy))
            with open(path_save, 'a', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                # write the data
                writer.writerow([name,epoch,accu_epoch,loss_epoch,mean_val_accuracy,mean_val_loss])
    return bestEpoch, bestLoss, bestAccuracy


def onlyVal(model,val_loader,device, loss_fn):
    model.eval()
    running_loss = 0.0 
    for name,loader in [("val", val_loader)]:
        correct= 0
        total = 0
        model.eval()
        with torch.no_grad():
            for imgs, labels in loader:

                imgs = imgs.to(device = device)
                labels = labels.to(device = device)
                outputs = model(imgs)
                loss = loss_fn(outputs, labels)
                running_loss +=  loss.item()

                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]                 
                correct += int((predicted == labels).sum())
        model.train()
        mean_val_accuracy = (100 * correct / total)               
        mean_val_loss = 100.*running_loss/len(val_loader) 
        return mean_val_accuracy,mean_val_loss

def validate(model, train_loader, val_loader, device):
    model.train()

    for name, loader in [("train", train_loader), ("val", val_loader)]:
        correct = 0
        correctSpin = 0
        totalSpin = 0
        # correctNotSpin = 0
        # totalNotSpin = 0
        correctUndetected = 0
        totalUndetected = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in loader:

                imgs = imgs.to(device = device)
                labels = labels.to(device = device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0] 
                for i in range(labels.size(dim=-1) ):
                    if(labels[i]==0):
                        if(labels[i]==predicted[i]):
                            correctSpin +=1
                        totalSpin +=1

                    # if(labels[i]==1):
                    #     if(labels[i]==predicted[i]):
                    #         correctNotSpin +=1
                    #     totalNotSpin +=1

                    if(labels[i]==1):
                        if(labels[i]==predicted[i]):
                            correctUndetected +=1
                        totalUndetected +=1

                                 
                correct += int((predicted == labels).sum())
        model.train()
        print("Accuracy {}: {:.2f}".format(name , correct / total))
        print("Accuracy Spin {}: {:.2f}".format(name , correctSpin / totalSpin))
        #print("Accuracy NoSpin {}: {:.2f}".format(name , correctNotSpin / totalNotSpin))
        print("Accuracy Undetected {}: {:.2f}".format(name , correctUndetected / totalUndetected))

def validateHeights(model,t,s,m,l,xl,device):
    for name, loader in [("tiny", t),("small", s),("medium", m),("large", l),("x-large", xl)]:
        correct = 0
        correctSpin = 0
        totalSpin = 0
        # correctNotSpin = 0
        # totalNotSpin = 0
        correctUndetected = 0
        totalUndetected = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in loader:

                imgs = imgs.to(device = device)
                labels = labels.to(device = device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0] 
                for i in range(labels.size(dim=-1) ):
                    if(labels[i]==0):
                        if(labels[i]==predicted[i]):
                            correctSpin +=1
                        totalSpin +=1

                    # if(labels[i]==1):
                    #     if(labels[i]==predicted[i]):
                    #         correctNotSpin +=1
                    #     totalNotSpin +=1

                    if(labels[i]==1):
                        if(labels[i]==predicted[i]):
                            correctUndetected +=1
                        totalUndetected +=1

                                    
                correct += int((predicted == labels).sum())
        print("Accuracy {}: {:.2f}".format(name , correct / total))
        print("Accuracy Spin {}: {:.2f}".format(name , correctSpin / totalSpin))
        #print("Accuracy NoSpin {}: {:.2f}".format(name , correctNotSpin / totalNotSpin))
        print("Accuracy Undetected {}: {:.2f}".format(name , correctUndetected / totalUndetected))


