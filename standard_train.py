from __future__ import print_function
import argparse  # Python 命令行解析工具
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # self.linear = nn.Linear(512 * block.expansion, num_classes)

        # --------------------------------------------------------
        self.fc1 = nn.Linear(512 * block.expansion, 3)
        self.fc2 = nn.Linear(3, num_classes)
        # --------------------------------------------------------

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)

        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, embed=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        if embed:
            return out
        out = self.fc2(out)
        return out


def get_dataset(args):
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('/youtu-face-identify-public/jiezhang/data/', train=True, download=True,
                         transform=transforms.Compose(
                             [
                                 transforms.RandomCrop(32, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                             ])),
        batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('/youtu-face-identify-public/jiezhang/data/', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])),
        batch_size=args.batch_size, shuffle=True, num_workers=4)
    return train_loader, test_loader


def train(model, train_loader, optimizer):
    model.train()
    total_loss = 0.0
    avg_loss = 0.0
    correct = 0.0
    acc = 0.0
    description = "loss={:.4f} acc={:.2f}%"

    with tqdm(train_loader) as epochs:
        for idx, (data, target) in enumerate(epochs):
            optimizer.zero_grad()
            epochs.set_description(description.format(avg_loss, acc))
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item()
            avg_loss = total_loss / (idx + 1)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc = correct / len(train_loader.dataset) * 100


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
            pred = torch.max(output, 1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print('\n Test_set: Average loss: {:.4f}, Accuracy: {:.4f}\n'
          .format(test_loss, acc))
    return acc, test_loss


def get_embeds(model, train_loader, type):
    model.eval()
    full_embeds = []
    full_labels = []
    if type == "tsne":
        nums = 40
    else:
        nums = 80
    with torch.no_grad():
        for i, (feats, labels) in enumerate(train_loader):
            feats = feats[:nums].cuda()
            full_labels.append(labels[:nums].cpu().detach().numpy())
            if type == "tsne":
                embeds = model(feats)
            else:
                embeds = model(feats, embed=True)
            full_embeds.append(F.normalize(embeds.detach().cpu()).numpy())
    return np.concatenate(full_embeds), np.concatenate(full_labels)


def plot(embeds, labels, fig_path='./example.pdf'):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create a sphere
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0 * pi:100j]
    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(phi)
    ax.plot_surface(
        x, y, z, rstride=1, cstride=1, color='w', alpha=0.3, linewidth=0)
    surf = ax.scatter(embeds[:, 0], embeds[:, 1], embeds[:, 2], c=labels, s=20, cmap=plt.get_cmap('rainbow'), )

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.grid(False)

    fig.colorbar(surf, fraction=0.1, shrink=1, aspect=5)
    plt.tight_layout()
    plt.savefig(fig_path)


def t_sne(latent_vecs, target):
    latent_vecs_reduced = TSNE(n_components=2, random_state=0).fit_transform(latent_vecs)
    plt.scatter(latent_vecs_reduced[:, 0], latent_vecs_reduced[:, 1],
                c=target, cmap='jet', s=20)  # s控制大小，默认为20
    plt.colorbar()
    plt.savefig('res.png')


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--load', type=int, default=0,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--type', type=str, default='tsne',
                        help='type of visualization')
    args = parser.parse_args()

    train_loader, test_loader = get_dataset(args)
    model = ResNet(BasicBlock, [2, 2, 2, 2]).cuda()

    if args.load == 1:
        model.load_state_dict(torch.load('model.pkl'))
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        bst_acc = -1
        for epoch in range(1, args.epochs + 1):
            train(model, train_loader, optimizer)
            acc, loss = test(model, test_loader)
            if acc > bst_acc:
                bst_acc = acc
                torch.save(model.state_dict(), 'model.pkl')

    embeds, labels = get_embeds(model, test_loader, args.type)
    if args.type == 'tsne':
        t_sne(embeds, labels)
    else:
        plot(embeds, labels, fig_path='baseline.png')


if __name__ == '__main__':
    main()

"""
python3 standard_train.py --lr=0.01 --batch_size=256 --load=0 --type=tsne
python3 standard_train.py --lr=0.01 --batch_size=256 --load=1 --type=sphere
"""
