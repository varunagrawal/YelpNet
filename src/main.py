#!/bin/python
from __future__ import print_function

import caffe
import numpy as np
import matplotlib.pyplot as plt
import json
import network
import solver
import utils

caffe.set_mode_gpu()
caffe.set_device(0)

caffe_root = "/home/varunagrawal/projects/caffe/"
dataset_root = "/home/varunagrawal/projects/YelpNet/data/"
images = dataset_root + "images/"

print("\n\n\n\n")
print("Loading Model")

#alex_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

#places_def = caffe_root + 'models/places205CNN/places205CNN_deploy.prototxt'
#places_weights = caffe_root + 'models/places205CNN/places205CNN_iter_300000.caffemodel'
places_weights = caffe_root + 'models/googlenet_places205/googlenet_places205_train_iter_2400000.caffemodel'

# You gave to generate this file using the attribute mapper utility
attributes_label_file = dataset_root + "business_attributes.json"
attributes_labels = json.load(open(attributes_label_file))

photo_2_business_json = dataset_root + "photo_id_to_business_id.json"
photo_2_business = json.load(open(photo_2_business_json))

# Setup the solver for our CNN
NUM_CLASSES = 65
batch_size = 128 # default is 128

train_net_file, test_net_file = "models/yelp_googlenet_train.prototxt", "models/yelp_googlenet_test.prototxt"
#train_net_file, test_net_file = solver.create_nets(NUM_CLASSES, batch_size)
sol_file = solver.create_solver(train_net_file, test_net_file, caffe_root)

# Finally time to fine-tune
nepochs = 10 #200

yelp_solver_filename = solver.create_solver(train_net_file, test_net_file, caffe_root=caffe_root)
yelp_solver = caffe.get_solver(yelp_solver_filename)
yelp_solver.net.copy_from(places_weights)


print('Running solvers for %d iterations...' % nepochs)
solvers = [('pretrained', yelp_solver),]
loss, acc, weights = solver.run_solvers(nepochs, solvers)
print('Done.')


#train_loss = loss['pretrained']
train_acc = acc['pretrained']
yelp_weights = weights['pretrained']

# Delete solvers to save memory.
del yelp_solver, solvers


#utils.plot(train_acc)
