#Load and normalize CIFAR10
import math

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib.ticker import MaxNLocator



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def loadAndNormalize():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    misclass_error = {classname: [] for classname in classes}
    # get some random training images
    #dataiter = iter(trainloader)
    #images, labels = dataiter.next()
    # show images
    #imshow(torchvision.utils.make_grid(images))
    # print labels
    #print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    #Train the network
    for epoch in range(3):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)
             #collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
                print("labellu",label)
                print(classes[label])

            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
                for classname in classes:
                    misclass_error[classname].append(1-(correct_pred[classname]/total_pred[classname]))
                correct_pred = {classname: 0 for classname in classes}
                total_pred = {classname: 0 for classname in classes}



    print('Finished Training')
    #Save model
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    #Test network
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    # print images
    #imshow(torchvision.utils.make_grid(images))
    #print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
    plot_misclassification_error(misclass_error,classes)
    #print(misclass_error)
    #return correct_pred,total_pred

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def plot_misclassification_error(misclass_error, classes):
    misclass_error={'plane': [0.8888888888888888, 0.5459976105137396, 0.5285359801488834, 0.5567901234567901, 0.5650510204081632, 0.5205992509363295, 0.5224171539961013, 0.47074122236671, 0.47757575757575754, 0.4527112232030265, 0.45399515738498786, 0.46143958868894597, 0.45333333333333337, 0.4135338345864662, 0.3880048959608323, 0.38819095477386933, 0.41002570694087404, 0.37231503579952263], 'car': [0.7744733581164808, 0.6586294416243654, 0.488833746898263, 0.41454545454545455, 0.3992042440318302, 0.35351089588377727, 0.37739872068230274, 0.32798165137614677, 0.34029484029484025, 0.31546894031668693, 0.3291614518147684, 0.32849604221635886, 0.32567849686847594, 0.2831541218637993, 0.29097839898348155, 0.2642679900744417, 0.24969097651421512, 0.3025641025641026], 'bird': [0.9911949685534591, 0.9475753604193972, 0.9082802547770701, 0.8041362530413625, 0.7409200968523002, 0.6723926380368098, 0.6859504132231404, 0.6936813186813187, 0.6322815533980582, 0.6105882352941177, 0.5879574970484062, 0.6417525773195876, 0.6006006006006006, 0.5798212005108556, 0.5325953259532595, 0.5602484472049689, 0.5695284159613059, 0.5564102564102564], 'cat': [0.9779141104294479, 0.8565989847715736, 0.7920792079207921, 0.7880364109232769, 0.7125307125307125, 0.7215346534653465, 0.704183266932271, 0.7056504599211564, 0.6734177215189874, 0.6364764267990075, 0.6583229036295369, 0.6329268292682927, 0.672244094488189, 0.597799511002445, 0.5924112607099143, 0.6381909547738693, 0.6456692913385826, 0.623574144486692], 'deer': [0.9402035623409669, 0.7883211678832117, 0.6385255648038051, 0.715561224489796, 0.7060333761232349, 0.6480099502487562, 0.6160896130346232, 0.561811505507956, 0.5746835443037974, 0.5896414342629481, 0.5587144622991347, 0.5121359223300971, 0.5244487056567593, 0.5167095115681234, 0.49877149877149873, 0.4733502538071066, 0.45229244114002476, 0.4942233632862645], 'dog': [0.7479289940828402, 0.6884236453201971, 0.6435246995994659, 0.564042303172738, 0.6002522068095839, 0.667574931880109, 0.543127962085308, 0.520496894409938, 0.5467349551856594, 0.5589005235602094, 0.5851197982345524, 0.5433455433455434, 0.55, 0.5157766990291262, 0.5874213836477987, 0.4903846153846154, 0.5460440985732815, 0.5406091370558376], 'frog': [0.5610047846889952, 0.5379825653798256, 0.49682337992376113, 0.4163473818646233, 0.38170731707317074, 0.4196891191709845, 0.36227544910179643, 0.3724747474747475, 0.3471177944862155, 0.3457106274007683, 0.3509852216748769, 0.3401442307692307, 0.31677018633540377, 0.3287671232876712, 0.32788944723618085, 0.33967046894803554, 0.3125, 0.3111931119311193], 'horse': [0.9422283356258597, 0.6443037974683544, 0.5431472081218274, 0.4651711924439197, 0.4528301886792453, 0.432657926102503, 0.39824732229795523, 0.4212454212454212, 0.41687657430730474, 0.385678391959799, 0.4042272126816381, 0.3740831295843521, 0.36820925553319916, 0.37250000000000005, 0.3732833957553059, 0.3707865168539326, 0.366504854368932, 0.34827144686299616], 'ship': [0.7599009900990099, 0.6764346764346765, 0.6242197253433208, 0.49538866930171277, 0.4384707287933094, 0.43003851091142486, 0.41058941058941056, 0.36687116564417177, 0.3746770025839793, 0.3487241798298907, 0.31474597273853777, 0.35240572171651496, 0.311623246492986, 0.32930107526881724, 0.3205619412515964, 0.29753086419753083, 0.30212234706616725, 0.29761904761904767], 'truck': [0.7696078431372549, 0.6169665809768637, 0.5693606755126659, 0.548, 0.4887218045112782, 0.45985401459854014, 0.4343029087261785, 0.46228710462287104, 0.4037037037037037, 0.44157441574415746, 0.43808255659121176, 0.3722084367245657, 0.3425309229305423, 0.37914110429447856, 0.3307593307593307, 0.34620334620334625, 0.35652173913043483, 0.33990147783251234]}
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    plt.clf()
    x=[]
    #print(misclass_error)
    for number in range(0,len(misclass_error['car'])):
        x.append((number+1)*2)
    print(x)
    for classname in classes:
        plt.plot(range(len(x)),misclass_error[classname])

    plt.legend(classes)
    plt.xlabel("trainer examples times for 1000")
    plt.ylabel("miscassification rate")

    plt.xticks(range(len(misclass_error['car'])), x)
    plt.title("Misclassification rate :")

    plt.show()

def main():
    #loadAndNormalize()
    #loadAndNormalize()
    #plot_misclassification_error(0,0)

    net = Net()
    PATH = './cifar_net.pth'
    net.load_state_dict(torch.load(PATH))
    batch_size = 4
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    dataiter = iter(testloader)
    images, labels = dataiter.next()
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    outputs = net(images)
    correct = 0
    total = 0
    #classLabelled= {classname:i for i,classname in enumerate(classes)}

    confusion_matrix = np.zeros((10,10))
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for label, prediction in zip(labels, predicted):
                confusion_matrix[label][prediction]+=1
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    print(confusion_matrix)


if __name__ == "__main__":
    main()



