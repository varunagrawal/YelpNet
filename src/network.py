from __future__ import print_function

import caffe
from caffe import layers as L
from caffe import params as P

weight_param = dict(lr_mult=1, decay_mult=1)
bias_param   = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]

frozen_param = [dict(lr_mult=0)] * 2


def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1,
              param=learned_param,
              weight_filler=dict(type='gaussian', std=0.01),
              bias_filler=dict(type='constant', value=0.1)):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, group=group,
                         param=param, weight_filler=weight_filler,
                         bias_filler=bias_filler)
    return conv, L.ReLU(conv, in_place=True)


def fc_relu(bottom, nout, param=learned_param,
            weight_filler=dict(type='gaussian', std=0.005),
            bias_filler=dict(type='constant', value=0.1)):
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
                        weight_filler=weight_filler,
                        bias_filler=bias_filler)
    return fc, L.ReLU(fc, in_place=True)


def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


def varnet(data_layer_params, datalayer, label=None, train=True, num_classes=65,
             classifier_name='fc8', learn_all=False):
    """
    Returns a NetSpec specifying VarNet, following the Places AlexNet proto text specification.
    Refer to: http://nbviewer.jupyter.org/github/BVLC/caffe/blob/tutorial/examples/03-fine-tuning.ipynb
    and for multilabel, refer: http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/pascal-multilabel-with-datalayer.ipynb
    """
    n = caffe.NetSpec()

    # Specify the data layer because we are doing mutlilabel classification
    n.data, n.label = L.Python(module='yelp_multilabel_datalayers',
                               layer=datalayer, ntop=2,
                               param_str=str(data_layer_params))

    print(n.label)

    param = learned_param if learn_all else frozen_param

    n.conv1, n.relu1 = conv_relu(n.data, 11, 96, stride=4, param=param)
    n.pool1 = max_pool(n.relu1, 3, stride=2)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
    n.conv2, n.relu2 = conv_relu(n.norm1, 5, 256, pad=2, group=2, param=param)
    n.pool2 = max_pool(n.relu2, 3, stride=2)
    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
    n.conv3, n.relu3 = conv_relu(n.norm2, 3, 384, pad=1, param=param)
    n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2, param=param)
    n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2, param=param)
    n.pool5 = max_pool(n.relu5, 3, stride=2)
    n.fc6, n.relu6 = fc_relu(n.pool5, 4096, param=param)

    ######################################
    # This is from the fine tuning example
    #if train:
    #    n.drop6 = fc7input = L.Dropout(n.relu6, in_place=True)
    #else:
    #    fc7input = n.relu6
    #n.fc7, n.relu7 = fc_relu(fc7input, 4096, param=param)
    #
    #if train:
    #    n.drop7 = fc8input = L.Dropout(n.relu7, in_place=True)
    #else:
    #    fc8input = n.relu7
    ########################################

    n.drop6 = L.Dropout(n.relu6, in_place=True)
    n.fc7, n.relu7 = fc_relu(n.drop6, 4096, param=param)
    n.drop7 = fc8input = L.Dropout(n.relu7, in_place=True)

    # always learn fc8 (param=learned_param)
    # renamed fc8 to n.score
    n.score = L.InnerProduct(fc8input, num_output=num_classes, param=learned_param)

    # This layer helps to do multilabel classification
    #n.loss = L.SigmoidCrossEntropyLoss(n.score, n.label, loss_weight=100)

    # This is a custom layer that gives squared error loss
    n.loss = L.Python(n.score, n.label,
                      module='multilabel_loss',
                      layer="MultiLabelLossLayer", ntop=1,
                      param_str=str({'weight':100}))

    proto = str(n.to_proto)
    # write the net to a file and return its prototxt and filename
    filename = "yelp_{0}.prototxt".format(data_layer_params['split'])
    with open(filename, 'w') as f:
        f.write(str(n.to_proto()))
        #print("Net written to {}".format(n.to_proto()))

    return proto, filename
