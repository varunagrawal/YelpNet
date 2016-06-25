# imports
import json
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


class YelpMultilabelDataLayer(caffe.Layer):

    """
    This is a simple syncronous datalayer for training a multilabel model on
    Yelp.
    """

    def setup(self, bottom, top):

        self.top_names = ['data', 'label']

        # === Read input parameters ===

        # params is a python dictionary with layer parameters.
        params = eval(self.param_str)

        # Check the parameters for validity.
        check_params(params)

        # store input as class variables
        self.batch_size = params['batch_size']

        self.batch_no = 0

        # Create a batch loader to load the images.
        self.batch_loader = BatchLoader(params, None)

        # === reshape tops ===
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.
        top[0].reshape(
            self.batch_size, 3, params['im_shape'][0], params['im_shape'][1])
        # Note the n_classes channels (because we have that many classes/labels.)
        top[1].reshape(self.batch_size, params['n_classes'])

        print_info("YelpMultilabelDataLayerSync", params)

    def forward(self, bottom, top):
        """
        Load data.
        """
        self.batch_no += 1
        print("Feed forward for batch {}".format(self.batch_no))

        for itt in range(self.batch_size):
            # Use the batch loader to load the next image.
            import datetime

            start = datetime.datetime.now()

            im, multilabel = self.batch_loader.load_next_image()

            #print datetime.datetime.now() - start

            # Add directly to the caffe data layer
            top[0].data[itt, ...] = im
            top[1].data[itt, ...] = multilabel

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        pass

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass


class BatchLoader(object):

    """
    This class abstracts away the loading of images.
    Images can either be loaded singly, or in a batch. The latter is used for
    the asyncronous data layer to preload batches while other processing is
    performed.
    """

    def __init__(self, params, result):
        self.result = result
        self.batch_size = params['batch_size']
        self.data_root = params['data_root']
        self.im_shape = params['im_shape']
        self.n_classes = params['n_classes']
        # get list of image indexes.
        list_file = params['split'] + '.json'

        # indexlist is a list of all the photo_ids of the Yelp dataset
        self.indexlist = [d['photo_id'] for d in json.load(open(osp.join(self.data_root, list_file)))]
        # self.indexlist = [line.rstrip('\n') for line in
        #    open(osp.join(self.data_root, 'ImageSets/Main', list_file))]
        self._cur = 0  # current image

        # Get the corresponding attributes for the image file
        # self.attrs = self.load_yelp_attributes(self.data_root, self.n_classes)
        self.load_yelp_attributes(self.data_root, self.n_classes)

        # this class does some simple data-manipulations
        self.transformer = SimpleTransformer()

        print "BatchLoader initialized with {} images".format(
            len(self.indexlist))

    def load_next_image(self):
        """
        Load the next image in a batch.
        """
        # Did we finish an epoch?
        if self._cur == len(self.indexlist):
            self._cur = 0
            shuffle(self.indexlist)

        #print "Loading image #{}".format(self._cur)

        # Load an image
        index = self.indexlist[self._cur]  # Get the image index
        image_file_name = index + '.jpg'
        im = np.asarray(Image.open(
            osp.join(self.data_root, 'images', image_file_name)))
        im = scipy.misc.imresize(im, self.im_shape)  # resize

        # do a simple horizontal flip as data augmentation
        flip = np.random.choice(2)*2-1
        im = im[:, ::flip, :]

        # Load and prepare ground truth
        multilabel = np.zeros(self.n_classes).astype(np.float32)

        # retrieve the attributes of the business to which this photo belongs
        attrs = self.get_business_attributes(index)
        # Always sort dict and return list of sorted key-value pairs to ensure feature vector is in always the same order
        attrs_list = sorted(attrs.iteritems())

        # print attrs
        for i, label in enumerate(attrs):
            # in the multilabel problem we don't care how MANY instances
            # there are of each class. Only if they are present.
            # The "-1" is b/c we are not interested in the background
            # class.
            multilabel[i] = attrs[label]

        self._cur += 1
        return self.transformer.preprocess(im), multilabel


    def load_yelp_attributes(self, data_root, n_classes):
        """
        This code is borrowed from Ross Girshick's FAST-RCNN code
        (https://github.com/rbgirshick/fast-rcnn).

        See publication for further details: (http://arxiv.org/abs/1504.08083).

        Thanks Ross!

        """
        classes = ("Accepts Credit Cards", "Alcohol",
        "Ambience_casual", "Ambience_classy", "Ambience_divey", "Ambience_hipster",
        "Ambience_intimate", "Ambience_romantic", "Ambience_touristy", "Ambience_trendy",
        "Ambience_upscale", "Attire_casual", "Attire_dressy", "Attire_formal", "Caters",
        "Delivery", "Dietary_Restrictions_dairy-free", "Dietary_Restrictions_gluten-free",
        "Dietary_Restrictions_halal", "Dietary_Restrictions_kosher", "Dietary_Restrictions_soy-free",
        "Dietary_Restrictions_vegan", "Dietary_Restrictions_vegetarian", "Drive-Thru",
        "Good_For_breakfast", "Good_For_brunch", "Good_For_dessert", "Good_For_dinner",
        "Good_For_latenight", "Good_For_lunch", "Good For Dancing", "Good For Groups",
        "Good for Kids", "Happy Hour", "Has TV", 'Music_background_music',
        'Music_dj', 'Music_jukebox', 'Music_karaoke', 'Music_live', 'Music_video',
        "Noise_Level_average", "Noise_Level_loud", "Noise_Level_quiet", "Noise_Level_very_loud",
        "Outdoor Seating",
        "Parking_garage", "Parking_lot", "Parking_street", "Parking_valet", "Parking_validated",
        "Price_Range_1", "Price_Range_2", "Price_Range_3", "Price_Range_4",
        'Smoking_outdoor', 'Smoking_yes', 'Smoking_no',
        "Take-out",
        "Takes Reservations",
        "Waiter Service",)
        class_to_ind = dict(zip(classes, xrange(n_classes)))

        attr_filename = osp.join(data_root, "business_attributes.json")
        # print 'Loading: {}'.format(filename)

        # Load the list of business attributes
        self.business_attributes = json.load(open(attr_filename))

        gt_classes = np.zeros((n_classes), dtype=np.int32)
        self.photo_2_business = json.load( open(osp.join(data_root, "photo_id_to_business_id.json")) )

        return self.business_attributes


    def get_business_attributes(self, index):
        """
        Get the attributes JSON for the image with name `index`
        """
        # Look for the business which the photo_2_business json points the image to.
        business = [b for b in self.photo_2_business if b["photo_id"] == index][0]

        #print sorted(self.business_attributes[0].keys())

        # Given the business from photo_2_business, use the business_id to find the attributes of the business
        attrs = [a for a in self.business_attributes if a["id"] == business["business_id"]][0]

        attributes = dict(attrs)

        # remove the id from the list of attributes
        attributes.pop("id")

        return attributes


def check_params(params):
    """
    A utility function to check the parameters for the data layers.
    """
    assert 'split' in params.keys(
    ), 'Params must include split (train, val, or test).'

    required = ['batch_size', 'data_root', 'im_shape']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)


def print_info(name, params):
    """
    Ouput some info regarding the class
    """
    print "{} initialized for split: {}, with bs: {}, im_shape: {}.".format(
        name,
        params['split'],
        params['batch_size'],
        params['im_shape'])
