# referenced from http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb

import caffe
import json
import tools
from PIL import Image
import scipy
import numpy as np
import sys


def run_net(net, image_file, truth, attributes, op_layer='loss3/classifierx'):
    """
    This code was mostly picked up from 
    http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/pascal-multilabel-with-datalayer.ipynb
    
    @image_file: This is the path to the image which will be run through the network
    @truth: This is the ground truth of the labels
    @attributes: This is a list of attributes we are considering. We follow the same ordering as in this list everywhere.
    @op_layer: The name of the output layer from which we get the outputs
    """
    image = Image.open(image_file)
    image = scipy.misc.imresize(image, (224, 224))  # resize

    transformer = tools.SimpleTransformer()
    net.blobs['data'].data[0, ...] = transformer.preprocess(image)

    output = net.forward()
    prob = net.blobs[op_layer].data[0, ...] > 0
    
    # print("Predictions are: {0}".format(prob))
    # print("Ground truth is: {0}".format(truth))
    
    true = [attributes[ind] for ind in range(len(attributes)) if truth[ind]]
    predicted = [attributes[ind] for ind in range(len(attributes)) if prob[ind]]

    if prob.size != truth.size:
        raise Exception("Prediction and Ground Truth labels are not of similar size")

    # Score calculated as per http://arxiv.org/pdf/1502.05988.pdf
    numer = 0
    denom = 0
    for ind in range(len(attributes)):
        numer = numer + (int(prob[ind]) * int(truth[ind]))
        denom = denom + (int(prob[ind]) + int(truth[ind]))

    if denom == 0:
        acc = 0
    else:
        acc = float(numer) / denom

    print acc

    photo_id = image_file.split('/')[-1]
    print(photo_id)
    print("Truth is: {0}".format(true))
    print("Predictions are: {0}".format(predicted))
    # scipy.misc.imshow(image)

    return acc


def main():
    # Load the parameters from the commandline
    try:
        model = sys.argv[1]
        weights = sys.argv[2]
        op_layer = sys.argv[3]
    except:
        print("Command should be:")
        print("python2 results.py <the model prototxt> <the model's weights> <the layer name which is the output layer, i.e. before the loss layer>")
        return
    
    # Set caffe to use the GPU
    caffe.set_mode_gpu()
    caffe.set_device(0)
    
    # Load all the test files
    p2b = json.load(open("../data/test.json"))

    # load the business attributes file
    b_attributes = json.load(open("../data/business_attributes.json", 'r'))

    # Load the Deep Network in Test mode
    # net = caffe.Net('models/yelp_googlenet_train_val.prototxt', 'snapshots/_iter_10000.caffemodel', caffe.TEST)
    net = caffe.Net(model, weights, caffe.TEST)

    # Create a list of attributes for back-pointing the attributes to
    attributes_list = []
    # Read the list of attributes from the file
    with open("../data/attributes_list") as attrs_list:
        attributes_list = attrs_list.readlines()
    attributes_list = [line.strip() for line in attributes_list]
    
    score = 0

    N_TEST = len(p2b)
    
    if len(sys.argv) > 4:    
        #N_TEST = 150
        N_TEST = sys.argv[3]

    for d in p2b[0:N_TEST]:
        b_id = d['business_id']
        photo_id = d['photo_id']

        # create a copy of the attribute dict since we delete entries from the copy
        # attr should be a dictionary of attributes mapped to 1 or 0
        attr = dict(next(attr for attr in b_attributes if attr['id'] == b_id))

        truth = np.zeros(len(attributes_list))

        del(attr['id'])
        for ind, a in enumerate(attributes_list):            
            truth[ind] = attr[a]
        
        # print(attr)
        # print(truth)
        image_file = "../data/images/" + photo_id + ".jpg"
        
        score = score + run_net(net, image_file, truth, attributes_list, op_layer)

    print("Labelling score: {0}".format(float(score) / N_TEST))


if __name__ == "__main__":
    main()
