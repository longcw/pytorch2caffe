## PyTorch2Caffe

Ported from [pytorch-caffe-darknet-convert](https://github.com/marvis/pytorch-caffe-darknet-convert). 

Add support for 
+ Dilated Convolution Layer
+ Concat Layer
+ Upsampling (converted to Deconvolution with bilinear initialization)
+ Eltwise Product
+ Sigmoid Layer

```python
# We can obtain almost the same output from caffe except Upsampling
# for inception_v3: 
# diff between pytorch and caffe: min: 0.0, max: 1.76429748535e-05, mean: 2.14079022953e-06
# see more in demo.py

import torch
from torch.autograd import Variable
import torchvision

import os
from pytorch2caffe import pytorch2caffe, plot_graph

m = torchvision.models.inception_v3(pretrained=True, transform_input=False)
m.eval()
print(m)

input_var = Variable(torch.rand(1, 3, 299, 299))
output_var = m(input_var)

output_dir = 'demo'
# plot graph to png
plot_graph(output_var, os.path.join(output_dir, 'inception_v3.dot'))

pytorch2caffe(input_var, output_var, 
              os.path.join(output_dir, 'inception_v3-pytorch2caffe.prototxt'),
              os.path.join(output_dir, 'inception_v3-pytorch2caffe.caffemodel'))

```

