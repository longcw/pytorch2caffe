import sys

sys.path.append('/data/build_caffe/caffe_rtpose/python')
import caffe
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable
from prototxt import *
import pydot

layer_dict = {'ConvNdBackward': 'Convolution',
              'ThresholdBackward': 'ReLU',
              'MaxPool2dBackward': 'Pooling',
              'AvgPool2dBackward': 'Pooling',
              'DropoutBackward': 'Dropout',
              'AddmmBackward': 'InnerProduct',
              'BatchNormBackward': 'BatchNorm',
              'AddBackward': 'Eltwise',
              'ViewBackward': 'Reshape',}

layer_id = 0


def pytorch2caffe(input_var, output_var, protofile, caffemodel):
    global layer_id
    net_info = pytorch2prototxt(input_var, output_var)
    print_prototxt(net_info)
    save_prototxt(net_info, protofile)

    net = caffe.Net(protofile, caffe.TEST)
    params = net.params

    layer_id = 1
    seen = set()

    def convert_layer(func):
        if True:
            global layer_id
            parent_type = str(type(func).__name__)

            if hasattr(func, 'next_functions'):
                for u in func.next_functions:
                    if u[0] is not None:
                        child_type = str(type(u[0]).__name__)
                        child_name = child_type + str(layer_id)
                        if child_type != 'AccumulateGrad' and (
                                parent_type != 'AddmmBackward' or child_type != 'TransposeBackward'):
                            if u[0] not in seen:
                                convert_layer(u[0])
                                seen.add(u[0])
                            if child_type != 'ViewBackward':
                                layer_id = layer_id + 1

            parent_name = parent_type + str(layer_id)
            print('converting %s' % parent_name)
            if parent_type == 'ConvNdBackward':
                weights = func.next_functions[1][0].variable.data
                if func.next_functions[2][0]:
                    biases = func.next_functions[2][0].variable.data
                else:
                    biases = None
                save_conv2caffe(weights, biases, params[parent_name])
            elif parent_type == 'BatchNormBackward':
                running_mean = func.running_mean
                running_var = func.running_var
                # print('%s running_mean' % parent_name, running_mean)
                # exit(0)
                scale_weights = func.next_functions[1][0].variable.data
                scale_biases = func.next_functions[2][0].variable.data
                bn_name = parent_name + "_bn"
                scale_name = parent_name + "_scale"
                save_bn2caffe(running_mean, running_var, params[bn_name])
                save_scale2caffe(scale_weights, scale_biases, params[scale_name])
            elif parent_type == 'AddmmBackward':
                biases = func.next_functions[0][0].variable.data
                weights = func.next_functions[2][0].next_functions[0][0].variable.data
                save_fc2caffe(weights, biases, params[parent_name])

    convert_layer(output_var.grad_fn)
    print('save caffemodel to %s' % caffemodel)
    net.save(caffemodel)


def save_conv2caffe(weights, biases, conv_param):
    if biases:
        conv_param[1].data[...] = biases.numpy()
    conv_param[0].data[...] = weights.numpy()


def save_fc2caffe(weights, biases, fc_param):
    print biases.size(), weights.size()
    print fc_param[1].data.shape
    print fc_param[0].data.shape
    fc_param[1].data[...] = biases.numpy()
    fc_param[0].data[...] = weights.numpy()


def save_bn2caffe(running_mean, running_var, bn_param):
    bn_param[0].data[...] = running_mean.numpy()
    bn_param[1].data[...] = running_var.numpy()
    bn_param[2].data[...] = np.array([1.0])


def save_scale2caffe(weights, biases, scale_param):
    scale_param[1].data[...] = biases.numpy()
    scale_param[0].data[...] = weights.numpy()


# def pytorch2prototxt(model, x, var):
def pytorch2prototxt(input_var, output_var):
    global layer_id
    net_info = OrderedDict()
    props = OrderedDict()
    props['name'] = 'pytorch'
    props['input'] = 'data'
    props['input_dim'] = input_var.size()

    layers = []

    layer_id = 1
    seen = set()
    top_names = dict()

    def add_layer(func):
        global layer_id
        parent_type = str(type(func).__name__)
        parent_bottoms = []

        if hasattr(func, 'next_functions'):
            for u in func.next_functions:
                if u[0] is not None:
                    child_type = str(type(u[0]).__name__)
                    child_name = child_type + str(layer_id)
                    if child_type != 'AccumulateGrad' and (
                            parent_type != 'AddmmBackward' or child_type != 'TransposeBackward'):
                        if u[0] not in seen:
                            top_name = add_layer(u[0])
                            parent_bottoms.append(top_name)
                            seen.add(u[0])
                        else:
                            top_name = top_names[u[0]]
                            parent_bottoms.append(top_name)
                        if child_type != 'ViewBackward':
                            layer_id = layer_id + 1

        parent_name = parent_type + str(layer_id)
        layer = OrderedDict()
        layer['name'] = parent_name
        layer['type'] = layer_dict[parent_type]
        parent_top = parent_name
        if len(parent_bottoms) > 0:
            layer['bottom'] = parent_bottoms
        else:
            layer['bottom'] = ['data']
        layer['top'] = parent_top
        if parent_type == 'ConvNdBackward':
            weights = func.next_functions[1][0].variable
            conv_param = OrderedDict()
            conv_param['num_output'] = weights.size(0)
            conv_param['pad'] = func.padding[0]
            conv_param['kernel_size'] = weights.size(2)
            conv_param['stride'] = func.stride[0]
            if func.next_functions[2][0] == None:
                conv_param['bias_term'] = 'false'
            layer['convolution_param'] = conv_param
        elif parent_type == 'BatchNormBackward':
            bn_layer = OrderedDict()
            bn_layer['name'] = parent_name + "_bn"
            bn_layer['type'] = 'BatchNorm'
            bn_layer['bottom'] = parent_bottoms
            bn_layer['top'] = parent_top
            batch_norm_param = OrderedDict()
            batch_norm_param['use_global_stats'] = 'true'
            bn_layer['batch_norm_param'] = batch_norm_param

            scale_layer = OrderedDict()
            scale_layer['name'] = parent_name + "_scale"
            scale_layer['type'] = 'Scale'
            scale_layer['bottom'] = parent_top
            scale_layer['top'] = parent_top
            scale_param = OrderedDict()
            scale_param['bias_term'] = 'true'
            scale_layer['scale_param'] = scale_param
        elif parent_type == 'ThresholdBackward':
            parent_top = parent_bottoms[0]
        elif parent_type == 'MaxPool2dBackward':
            pooling_param = OrderedDict()
            pooling_param['pool'] = 'MAX'
            pooling_param['kernel_size'] = func.kernel_size[0]
            pooling_param['stride'] = func.stride[0]
            # http://netaz.blogspot.com/2016/08/confused-about-caffes-pooling-layer.html
            padding = 0 if func.padding[0] in {0, 1} else func.padding[0]
            pooling_param['pad'] = padding
            layer['pooling_param'] = pooling_param
        elif parent_type == 'AvgPool2dBackward':
            pooling_param = OrderedDict()
            pooling_param['pool'] = 'AVE'
            pooling_param['kernel_size'] = func.kernel_size[0]
            pooling_param['stride'] = func.stride[0]
            layer['pooling_param'] = pooling_param
        elif parent_type == 'DropoutBackward':
            parent_top = parent_bottoms[0]
            dropout_param = OrderedDict()
            dropout_param['dropout_ratio'] = func.p
            layer['dropout_param'] = dropout_param
        elif parent_type == 'AddmmBackward':
            inner_product_param = OrderedDict()
            inner_product_param['num_output'] = func.next_functions[0][0].variable.size(0)
            layer['inner_product_param'] = inner_product_param
        elif parent_type == 'ViewBackward':
            parent_top = parent_bottoms[0]
        elif parent_type == 'AddBackward':
            eltwise_param = OrderedDict()
            eltwise_param['operation'] = 'SUM'
            layer['eltwise_param'] = eltwise_param

        layer['top'] = parent_top  # reset layer['top'] as parent_top may change
        if parent_type != 'ViewBackward':
            if parent_type == "BatchNormBackward":
                layers.append(bn_layer)
                layers.append(scale_layer)
            else:
                layers.append(layer)
                # layer_id = layer_id + 1
        top_names[func] = parent_top
        return parent_top

    add_layer(output_var.grad_fn)
    net_info['props'] = props
    net_info['layers'] = layers
    return net_info


def plot_graph(top_var, fname, params=None):
    """
    This method don't support release v0.1.12 caused by a bug fixed in: https://github.com/pytorch/pytorch/pull/1016
    So if you want to use `plot_graph`, you have to build from master branch or wait for next release.

    Plot the graph. Make sure that require_grad=True and volatile=False
    :param top_var: network output Varibale
    :param fname: file name
    :param params: dict of (name, Variable) to add names to node that
    :return: png filename
    """
    from graphviz import Digraph
    import pydot
    dot = Digraph(comment='LRP',
                  node_attr={'style': 'filled', 'shape': 'box'})
    # , 'fillcolor': 'lightblue'})

    seen = set()

    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = '{}\n '.format(param_map[id(u)]) if params is not None else ''
                node_name = '{}{}'.format(name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(top_var.grad_fn)
    dot.save(fname)
    (graph,) = pydot.graph_from_dot_file(fname)
    im_name = '{}.png'.format(fname)
    graph.write_png(im_name)
    print(im_name)

    return im_name


if __name__ == '__main__':
    import torchvision

    m = torchvision.models.inception_v3(pretrained=True, transform_input=False)
    m.eval()  # very important here, otherwise batchnorm running_mean, running_var will be incorrect
    input_var = Variable(torch.rand(1, 3, 299, 299))

    print(m)
    output_var = m(input_var)
    plot_graph(output_var, 'inception_v3.dot')

    pytorch2caffe(input_var, output_var, 'inception_v3-pytorch2caffe.prototxt', 'inception_v3-pytorch2caffe.caffemodel')

    # test

