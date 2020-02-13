import torch
from PIL import Image
from pathlib import Path
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data
from torchvision import datasets, transforms
import argparse
import numpy as np

preprocess = transforms.Compose([ transforms.Resize(32), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
classesMapping = {
        0: "an airplane",
        1: "an automobile",
        2: "a bird",
        3: "a cat",
        4: "a deer",
        5: "a dog",
        6: "a frog",
        7: "an horse",
        8: "a ship",
        9: "a truck" }

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3, 1)
        self.conv2 = nn.Conv2d(128, 128, 3, 1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1)
        self.conv5 = nn.Conv2d(256, 256, 3, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128,10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.dropout1(x)
        #x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def main():
    parser = argparse.ArgumentParser(description="CNN predictions")
    parser.add_argument("--file", action="store", default='', metavar='I',
            help='Path of image to predict')
    parser.add_argument("--threshold", type=int, default=3, metavar='T', help='Threshold between the first predictions')
    args = parser.parse_args()
    s = args.threshold
    model = Net()
    model.load_state_dict(torch.load("cifar10_cnn.pt"))
    model.eval()
    image = Image.open(Path(args.file))
    if abs(image.size[1]-image.size[0])>100:
        d = min(image.size)
        image = image.crop((0,0,d,d))
    singleton = preprocess(image).unsqueeze_(0)
    #print(singleton.shape)
    output = model(singleton)
    #print(output)
    prediction = output.detach().numpy()[0]
    #print(prediction)
    c = np.argpartition(-prediction, (0, 2))
    if -(prediction[c[1]]-prediction[c[0]])>s:
        print("It's {}".format(classesMapping[c[0]]))
    else: 
        print("It's probably {}, but maybe it's {}".format(classesMapping[c[0]], classesMapping[c[1]]))

if __name__=="__main__":
    main()
