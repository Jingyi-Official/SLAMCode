'''
Description: 
Author: 
Date: 2022-09-19 21:49:23
LastEditTime: 2022-10-26 15:54:29
LastEditors: Jingyi Wan
Reference: 
'''
import cv2
import torchvision.transforms as transforms

class Image_Transforms(object):
    def __init__(self):
        self.image_transform = transforms.Compose(
            [BGRtoRGB(),
            NormRGB()])

    def __call__(self, image):
        return self.image_transform(image)

class BGRtoRGB(object):
    """bgr format to rgb"""
    def __call__(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

class NormRGB(object):
    def __call__(self, image):
        image = image.astype(float)/255.
        return image