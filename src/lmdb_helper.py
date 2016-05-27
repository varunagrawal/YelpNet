# Reference http://deepdish.io/2015/04/28/creating-lmdb-in-python/
# https://groups.google.com/forum/#!topic/caffe-users/RuT1TgwiRCo

import caffe
import lmdb
import sys
import numpy as np
from PIL import Image
import scipy
import json


n_classes = 64


def get_image_file_list(file_):
    """
    Read the train/test json, get the path to the image and the corresponding attributes for that image
    Labels should be list of (K, 1, 1) with K being the number of attributes
    Each label needs to have 3 dimensions in order for Caffe to be able to read it properly, hence we pad with singleton dimensions
    https://github.com/BVLC/caffe/blob/5166583b077ac9889ac7dabbd29bd879e21e0b70/python/caffe/io.py
    https://github.com/BVLC/caffe/issues/1698#issue-53768814

    File should be the path to the test or train json file e.g. ../data/train.json
    """
    image_files = []
    labels = []

    print('Reading images')
    p2b = json.load(open(file_))
    b_attributes = json.load(open("../data/business_attributes.json", 'r'))

    # Read the list of attributes from the file
    with open("../data/attributes_list") as attrs_list:
        attributes_list = attrs_list.readlines()
        
    for d in p2b:
        b_id = d['business_id']
        photo_id = d['photo_id']

        image_files.append("../data/images/" + photo_id + ".jpg")

        # create a copy of the attribute dict since we delete entries from the copy
        a = dict(next(attr for attr in b_attributes if attr['id'] == b_id))
        x = np.zeros((n_classes, 1, 1))

        # Remove the ID field as that is not an attribute
        del(a['id'])

        for i, k in enumerate(attributes_list:
            if not k == 'id':
                x[i, :, :] = a[k]

        labels.append(x)

        print(photo_id)

    return image_files, labels


def create_image_lmdb(images):
    print("Creating image LMDB")
    height = 224
    width = 224

    in_db = lmdb.open("../data/image-{0}-lmdb".format(split), map_size=int(1e12))
    with in_db.begin(write=True) as in_txn:
        for idx, in_ in enumerate(image_files):
            # load image:
            # - as np.uint8 {0, ..., 255}
            # - in BGR (switch from RGB)
            # - in Channel x Height x Width order (switch from H x W x C)
            image = Image.open(in_)  # or load whatever ndarray you need

            image = scipy.misc.imresize(image, (width, height))  # resize
            # image = image.resize((width, height))

            im = np.asarray(image)
            # scipy.misc.imshow(im)

            im = im[:, :, ::-1]
            im = im.transpose((2, 0, 1))

            im_data = caffe.io.array_to_datum(im)

            print("Saving {0}".format(in_))
            in_txn.put('{:0>10d}'.format(idx), im_data.SerializeToString())

    in_db.close()


def create_label_lmdb(labels_list):
    print("Creating label LMDB")

    in_db = lmdb.open("../data/label-{0}-lmdb".format(split), map_size=int(1e12))
    with in_db.begin(write=True) as in_txn:
        for idx, label in enumerate(labels_list):

            data = caffe.io.array_to_datum(label)

            print("Saving {0}".format(idx))
            in_txn.put('{:0>10d}'.format(idx), data.SerializeToString())

    in_db.close()


if len(sys.argv) < 3:
    print("Usage: python2 lmdb_helper.py path/to/photo_2_business_id.json [train|val|test]")
    exit()

print(sys.argv)

# Whether to create train, val or test LMDB
split = sys.argv[2]

image_files, labels = get_image_file_list("../data/{0}.json".format(split))

create_image_lmdb(image_files)
create_label_lmdb(labels)
