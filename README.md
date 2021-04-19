# visualization
This repository implements visualization of tsne and sphere features.

First of all, train a resnet-18.
```shell
python3 standard_train.py --lr=0.01 --batch_size=256 --load=0
```

## T-sne
```shell
python3 standard_train.py --lr=0.01 --batch_size=256 --load=1 --type=tsne
```
![image](https://github.com/devilzj/visualization/blob/main/tsne_cifar10.png)

## Sphere
```shell
python3 standard_train.py --lr=0.01 --batch_size=256 --load=1 --type=sphere
```
![image](https://github.com/devilzj/visualization/blob/main/sphere_cifar10.png)
