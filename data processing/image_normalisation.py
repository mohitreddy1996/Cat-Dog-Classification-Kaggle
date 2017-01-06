import os
import glob
import random
import numpy as np

import cv2

import caffe
from caffe.proto import caffe_pb2

import lmdb


# define the final height and width
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227


# histogram equalization for 3 channels.
def histogram_equalization(image):
    image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
    image[:, :, 1] = cv2.equalizeHist(image[:, :, 1])
    image[:, :, 2] = cv2.equalizeHist(image[:, :, 2])

    return image


# resize the image
def image_resizer(image, image_width, image_height):
    return cv2.resize(image, (image_width, image_height), interpolation=cv2.INTER_CUBIC)


# image tranform
def transform_image(image, image_width, image_height):
    image = histogram_equalization(image)
    return image_resizer(image, image_width, image_height)


# make datum : google protobuf message data type which can store data in the form of optinal label, data.
def make_datum(image, label, image_width, image_height):
    return caffe_pb2.Datum(channels=3, width=image_width, height=image_height, label=label,
                           data=np.rollaxis(image, 2).tostring())

# lmdb location
train_location_db = '/home/mohit/Cat-Dog-Classification-Kaggle/input/train_lmdb'
validation_location_db = '/home/mohit/Cat-Dog-Classification-Kaggle/input/val_lmdb'

# remove the previous data.
os.system('rm -rf ' + train_location_db)
os.system('rm -rf ' + validation_location_db)

# get the train data and test data.
train_data = [image for image in glob.glob("../input/train/*jpg")]
test_data = [image for image in glob.glob("../input/test1/*jpg")]

# randomly shuffle data
random.shuffle(train_data)

'''
    Data - Training Data (5/6th of Data), Validation Data (1/6th of Data)
'''

print " Starting with image Normalisation and storing in lmdb"

# begin transaction for training data
db_cur = lmdb.open(train_location_db, map_size=int(1e12))
with db_cur.begin(write=True) as txn:
    for idx, image_path in enumerate(train_data):
        if idx % 6 == 0:
            continue
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # get the transformed image (histogram equaliser and resizing)
        image = transform_image(image, IMAGE_WIDTH, IMAGE_HEIGHT)
        # put labels for the image
        if 'cat' in image_path:
            label = 0
        else:
            label = 1

        datum = make_datum(image, label, IMAGE_WIDTH, IMAGE_HEIGHT)
        txn.put('{:0>5d}'.format(idx), datum.SerializeToString())
        print '{:0>5d}'.format(idx) + " : " + image_path
db_cur.close()

print "Finished with Training data"

# begin transaction for validation data
db_cur = lmdb.open(validation_location_db, map_size=int(1e12))
with db_cur.begin(write=True) as txn:
    for idx, image_path in enumerate(train_data):
        if idx % 6 != 0:
            continue
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = transform_image(image, IMAGE_WIDTH, IMAGE_HEIGHT)
        if 'cat' in image_path:
            label = 0
        else:
            label = 1
        datum = make_datum(image, label, IMAGE_WIDTH, IMAGE_HEIGHT)
        txn.put('{:0>5d}'.format(idx), datum.SerializeToString())
        print '{:0>5d}'.format(idx) + " : " + image_path
db_cur.close()

print "\n Finished With all the images!"

