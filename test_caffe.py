import sys
sys.path.append('/data/build_caffe/caffe_rtpose/python')
import caffe

import numpy as np
import os
import cv2
import torch
import torchvision
from torch.autograd import Variable

caffe.set_device(0)
caffe.set_mode_gpu()

model_def = 'inception_v3-pytorch2caffe.prototxt'
model_weights = 'inception_v3-pytorch2caffe.caffemodel'

net = caffe.Net(model_def, model_weights, caffe.TEST)

test_image_name = 'dog.jpg'

img = cv2.imread(test_image_name, cv2.IMREAD_COLOR)
img = cv2.resize(img, (299, 299))

img = np.transpose(img, (2, 0, 1))
img = np.expand_dims(img, 0).astype(np.float32)
img[:, 0, :, :] -= 103.939
img[:, 1, :, :] -= 116.779
img[:, 2, :, :] -= 123.68

net.blobs['data'].data[...] = img

net.forward(start='ConvNdBackward1')

prediction = net.blobs['AddmmBackward348'].data

print(np.argmax(prediction), np.max(prediction))


m = torchvision.models.inception_v3(pretrained=True, transform_input=False).cuda()
m.eval()

input_var = Variable(torch.from_numpy(img.astype(np.float32))).cuda()

output_var = m(input_var)

pred_torch = output_var.data.cpu().numpy()

print(np.argmax(pred_torch), np.max(pred_torch))