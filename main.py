import torch
from tqdm import tqdm
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F

BATCH = 128
epochs = 3

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
trainset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH, shuffle=True)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x


# check if using gpu or cpu
if torch.cuda.is_available():
    dev = "cuda"
    print("USING CUDA")
else:
    dev = "cpu"
    print("USING CPU")

device = torch.device(dev)
net = LeNet()
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

for e in range(epochs):
    for images, labels in tqdm(trainloader):

        # convert data to cuda float tensor
        images = images.cuda()
        labels = labels.cuda()

        # set gradient to none
        optimizer.zero_grad(set_to_none=True)

        # forward prop then backward prop then take gradient step
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print('~~~~~~~~~~~Finished Training~~~~~~~~~~~~~~')

# test the accuracy of the system
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images = images.cuda()
        labels = labels.cuda()

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        # print(predicted)
        correct += (predicted == labels).sum().item()

print('Accuracy of net using test images set: %d %%' % (100 * correct / total))
