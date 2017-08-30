# -*- coding: utf-8
from pytorch2caffe import plot_graph, pytorch2caffe
import sys
sys.path.append('/data/build_caffe/caffe_rtpose/python')
import caffe
import numpy as np
import os

import torch
from torch.autograd import Variable
import torchvision


# test the model or generate model
test_mod = True

caffemodel_dir = 'demo'
input_size = (1, 3, 299, 299)

model_def = os.path.join(caffemodel_dir, 'model.prototxt')
model_weights = os.path.join(caffemodel_dir, 'model.caffemodel')
input_name = 'ConvNdBackward1'
output_name = 'AddmmBackward348'

# pytorch net
model = torchvision.models.inception_v3(pretrained=True, transform_input=False)
model.eval()

# random input
image = np.random.randint(0, 255, input_size)
input_data = image.astype(np.float32)

# pytorch forward
input_var = Variable(torch.from_numpy(input_data))

if not test_mod:
    # generate caffe model
    output_var = model(input_var)
    plot_graph(output_var, os.path.join(caffemodel_dir, 'pytorch_graph.dot'))
    pytorch2caffe(input_var, output_var, model_def, model_weights)
    exit(0)

# test caffemodel
caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net(model_def, model_weights, caffe.TEST)

net.blobs['data'].data[...] = input_data
net.forward(start=input_name)
caffe_output = net.blobs[output_name].data

model = model.cuda()
input_var = input_var.cuda()
output_var = model(input_var)
pytorch_output = output_var.data.cpu().numpy()

print(input_size, pytorch_output.shape, caffe_output.shape)
print('pytorch: min: {}, max: {}, mean: {}'.format(pytorch_output.min(), pytorch_output.max(), pytorch_output.mean()))
print('  caffe: min: {}, max: {}, mean: {}'.format(caffe_output.min(), caffe_output.max(), caffe_output.mean()))

diff = np.abs(pytorch_output - caffe_output)
print('   diff: min: {}, max: {}, mean: {}, median: {}'.format(diff.min(), diff.max(), diff.mean(), np.median(diff)))