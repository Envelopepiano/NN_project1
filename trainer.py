import numpy as np
import torch
import matplotlib.pyplot as plt

import sys
from data import saveModel

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../')

def trainer(net, criterion, optimizer, trainloader, devloader, epoch_n=10000, path="./checkpoint/save.pt"):
    train_losses = []  # 用於存儲訓練損失
    valid_losses = []  # 用於存儲驗證損失
    for epoch in range(epoch_n): # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        train_loss = 0.0
        valid_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, data_index = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs.cpu(), labels.unsqueeze(1).cpu())
            train_loss += loss.item()*inputs.shape[0]
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

        ######################
        # validate the model #
        ######################
        net.eval()
        for i, data in enumerate(devloader, 0):
            # move tensors to GPU if CUDA is available
            inputs, labels, data_index = data
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = net(inputs)
            # calculate the batch loss
            loss = criterion(outputs.cpu(), labels.cpu())
            # update average validation loss
            valid_loss += loss.item()*inputs.shape[0]

        # calculate average losses
        train_loss = train_loss/len(trainloader.dataset)
        valid_loss = valid_loss/len(devloader.dataset)

        # print training/validation statistics
        print('\tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(train_loss, valid_loss))

        # Append the losses to lists
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

    print('Finished Training')

    # Plot the training and validation loss curves
    plt.figure()
    plt.plot(range(epoch_n), train_losses, label='Training Loss')
    plt.plot(range(epoch_n), valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.show()

    ## Save model
    saveModel(net, path)
    
def tester(net, criterion, testloader):
    loss = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels, data_index = data
            outputs = net(inputs,labels)
            loss += criterion(outputs.cpu(), labels.unsqueeze(1).cpu())
    return loss.item()
