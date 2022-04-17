import datetime
import enum

import torch

def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, device): 
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device = device)
            labels = labels.to(device = device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {}, Training loss {}'.format(
            datetime.datetime.now(), epoch,
            loss_train / len(train_loader)))

def validate(model, train_loader, val_loader, device):
    for name, loader in [("train", train_loader), ("val", val_loader)]:
        correct = 0
        correctSpin = 0
        totalSpin = 0
        correctNotSpin = 0
        totalNotSpin = 0
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

                    if(labels[i]==1):
                        if(labels[i]==predicted[i]):
                            correctNotSpin +=1
                        totalNotSpin +=1

                    if(labels[i]==2):
                        if(labels[i]==predicted[i]):
                            correctUndetected +=1
                        totalUndetected +=1

                                 
                correct += int((predicted == labels).sum())
        print("Accuracy {}: {:.2f}".format(name , correct / total))
        print("Accuracy Spin {}: {:.2f}".format(name , correctSpin / totalSpin))
        print("Accuracy NoSpin {}: {:.2f}".format(name , correctNotSpin / totalNotSpin))
        print("Accuracy Undetected {}: {:.2f}".format(name , correctUndetected / totalUndetected))


