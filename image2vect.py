from keras.models import load_model
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot
from sklearn.preprocessing import normalize
import argparse
import sys
import os

from yoloface.utils import *

"""
FaceNet Model from:
https://github.com/nyoki-mtl/keras-facenet
Yolo Model and implementation adapted from:
https://github.com/sthanhng/yoloface
"""

#####################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--model-cfg', type=str, default='./yoloface/cfg/yolov3-face.cfg',
                    help='path to config file')
parser.add_argument('--model-weights', type=str,
                    default='./yoloface/model-weights/yolov3-wider_16000.weights',
                    help='path to weights of model')
parser.add_argument('--image', type=str, default='',
                    help='path to image file')
parser.add_argument('--video', type=str, default='',
                    help='path to video file')
parser.add_argument('--src', type=int, default=0,
                    help='source of the camera')
parser.add_argument('--output-dir', type=str, default='outputs/',
                    help='path to the output directory')
args = parser.parse_args()

#####################################################################

# load the facenet model
print("Loading FaceNet Model")
face_net_model = load_model('facenet_keras.h5')

print("Loading Yolo v3 Model")
net = cv2.dnn.readNetFromDarknet(args.model_cfg, args.model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

downscale_size = (160, 160)

# extract a single face from a given photograph
def image2vect(filename):

    cap = cv2.VideoCapture(filename)
    has_frame, frame = cap.read()
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                 [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    outs = net.forward(get_outputs_names(net))

    faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)

    # check if face was found
    if len(faces) == 0:
        # did not find face
        print("Cannot detect face for file!", filename)
        # return an embedding of all 100
        return np.full((1, 128), 100)

    # left top right bottom
    # print(faces)
    box_data = faces[0]
    left, top, right, bottom = refined_box(box_data[0], box_data[1], box_data[2], box_data[3])
    image = Image.open(filename)
    image = image.crop((left, top, right, bottom))
    face = np.asarray(image)
    image = Image.fromarray(face)
    image = image.resize(downscale_size)
    face_array = np.asarray(image)
    # pyplot.imshow(face_array)
    # pyplot.show()
    # print(face_array.shape)

    # change pixels to floats
    face_pixels = face_array.astype('float32')
    # standardize pixel values
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # print(face_pixels.shape)
    # prepare pixels for facenet
    face_pixels = np.expand_dims(face_pixels, axis=0)
    # print(face_pixels.shape)
    # use facenet to get embedding
    embedding_unnormalized = face_net_model.predict(face_pixels)
    embedding_unnormalized = embedding_unnormalized[0]
    # use l2 normalization
    embedding = normalize(embedding_unnormalized.reshape(1, -1), norm='l2')

    return embedding



# test the image2vect function
# path = 'selected_jpgs/052003.jpg' # cant recognize face for this pic
# # path = 'selected_jpgs/000109.jpg'
# embedding = image2vect(path)
# print(embedding)
#
# sum = 0
# for i in range(len(embedding[0])):
#     sum += pow(embedding[0][i], 2)
# print(sum)
