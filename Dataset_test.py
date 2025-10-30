import os
import numpy as np
import torch
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data.dataset import Dataset

    
class ToTensor(object):
    def __call__(self, samples):
        images, labels = samples['images'], samples['labels']
        images = (images.transpose((0,3,1,2)).astype('float32') / 255)
        return {'images': torch.from_numpy(images),
                'labels': torch.from_numpy(labels).long()}


class AugData():
    def __call__(self, samples):
        images, labels = samples['images'], samples['labels']

        # Rotation
        rot = random.randint(0, 3)
        images = np.rot90(images, rot, axes=[1, 2]).copy()
        # Mirroring
        if random.random() < 0.5:
            images = np.flip(images, axis=2).copy()

        new_sample = {'images': images, 'labels': labels}
        return new_sample
    

def build_Dataloader(args, dataset_dir):
    train_transform = transforms.Compose([
        AugData(),
        ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])


    test_dataset = ImageFolder(
        root=os.path.join(dataset_dir, 'test'),
        transform=test_transform,
    )


    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )


    return test_dataloader