import caffe
import numpy as np

class MultiLabelLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        # Setup top layers
        #self.top_names = ['multilabelloss',]

        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

        # params is a python dictionary with layer parameters.
        params = eval(self.param_str)

        # Set the loss weight for the layer
        self.weight = params['weight']



    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        # Calculate Euclidean distance
        self.diff[...] = bottom[0].data - bottom[1].data
        # Weigh the averaged Euclidean distance and set it to the data of this layer
        top[0].data[...] = self.weight * np.sum(self.diff**2) / bottom[0].num / 2.

    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff / bottom[i].num
