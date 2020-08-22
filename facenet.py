#!/usr/bin/python3
import os
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torchvision

class FaceNet:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.get_image = torchvision.transforms.ToPILImage()

    # calculate the euclidean distance of two face tensors: (batch, 3, height, width)
    def __call__(self, face1, face2):
        batch = face1.size(0)
        d = torch.zeros(batch, 1)
        for i in range(batch):
            a = self.get_image(face1[i].cpu().clamp(0,1))
            b = self.get_image(face2[i].cpu().clamp(0,1))
            f1 = self.crop(a).to(self.device)
            f2 = self.crop(b).to(self.device)
            embedding1 = self.resnet(f1.unsqueeze(0))
            embedding2 = self.resnet(f2.unsqueeze(0))
            d[i] = torch.cdist(embedding1, embedding2)
        # perserve the gradients?
        grad = (face1 + face2).sum()
        loss = grad - grad + d.sum()
        return loss

    def crop(self, image):
        cropped = self.mtcnn(image)
        # no face detected
        if cropped is None:
            cropped = torch.zeros(3, 20, 20)
        return cropped

# test
if False:
    x = torch.randn(1, 3, 160, 160)
    y = torch.randn(1, 3, 160, 160)
    facenet = FaceNet()
    d = facenet(x, y)
    d.backward()

