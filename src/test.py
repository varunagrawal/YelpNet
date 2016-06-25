import time
import pickle
import scipy.misc
import skimage.io
import caffe

import numpy as np
import os.path as osp

from xml.dom import minidom
import json
from random import shuffle
from threading import Thread
from PIL import Image

from tools import SimpleTransformer
import solver


caffe.set_mode_gpu()
caffe.set_device(0)

caffe_root = "/home/varunagrawal/projects/advanced_computer_vision/caffe/"


class Tester:

    def __init__(self):
        train_net_file = "yelp_train.prototxt"
        test_net_file = "yelp_test.prototxt"
        weights = "weights/weights.pretrained.caffemodel"

        yelp_solver_filename = "yelp_multilabel_solver.prototxt"
        yelp_solver = caffe.get_solver(yelp_solver_filename)
        yelp_solver.net.copy_from(weights)

        self.net = yelp_solver.test_nets[0]

    def run(self):
        acc = 0.0
        batch_size = 128
        num_batches = 1#57 # (20000 / batch_size) +1

        for t in range(num_batches):
            self.net.forward()
            gts = self.net.blobs['label'].data
            ests = self.net.blobs['score'].data > 0
            for gt, est in zip(gts, ests): #for each ground truth and estimated label vector
                acc += self.hamming_distance(gt, est)
            print acc / ((t+1) * batch_size)

        return acc / (num_batches * batch_size)

    def hamming_distance(self, gt, est):
        # Use Hamming Distance to find multilabel loss
        return sum([1 for (g, e) in zip(gt, est) if g == e]) / float(len(gt))

    def check_baseline_accuracy(self, num_batches=1, batch_size = 128):
        acc = 0.0
        net = self.net
        for t in range(num_batches):
            net.forward()
            gts = net.blobs['label'].data
            ests = np.zeros((batch_size, len(gts)))
            for gt, est in zip(gts, ests): #for each ground truth and estimated label vector
                acc += self.hamming_distance(gt, est)
        return acc / (num_batches * batch_size)


if __name__ == "__main__":
    tester = Tester()
    #print tester.check_baseline_accuracy()
    print tester.run()
