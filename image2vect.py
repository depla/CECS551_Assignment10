from keras.models import load_model
import os
from PIL import Image
import numpy as np
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot
from sklearn.preprocessing import normalize

"""
FaceNet Model from:
https://github.com/nyoki-mtl/keras-facenet
Code from:
https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/
using MTCNN library
"""

# load the facenet model
print("Loading FaceNet Model")
face_net_model = load_model('facenet_keras.h5')
downscale_size = (160, 160)

# extract a single face from a given photograph
def image2vect(filename):
    # load image from file
    image = Image.open(filename)
    # convert to array
    pixels = np.asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # check if a face was found
    if len(results) == 0:
        # did not find face
        print("Cannot detect face for file!", filename)
        # return an embedding of all 100
        return np.full((1, 128), 100)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(downscale_size)
    face_array = np.asarray(image)
    # pyplot.imshow(face_array)
    # pyplot.show()

    # scale pixel values
    face_pixels = face_array.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = np.expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    embedding_unnormalized = face_net_model.predict(samples)
    embedding_unnormalized = embedding_unnormalized[0]
    embedding = normalize(embedding_unnormalized.reshape(1, -1), norm='l2')

    return embedding



# test the image2vect function
# path = 'selected_jpgs/052003.jpg' # cant recognize face for this pic
# # path = 'selected_jpgs/000109.jpg'
# embedding = image2vect(path)
# print(embedding)


################################ FAILED YOLO STUFF #####################################
# couldn't get the YOLO from https://github.com/sthanhng/yoloface to cooperate.
# kept getting errors :(
# class Args:
#     def __init__(self):
#         self.model = 'YOLO_Face.h5'
#         self.classes = 'cfg/face_classes.txt'
#         self.anchors = 'cfg/yolo_anchors.txt'
#         self.img_size = (416, 416)
#         self.score = 0.5
#         self.iou = 0.5
#
#
# path = 'selected_jpgs/000109.jpg'
#
# args = Args()
# yolo_model = yolo.YOLO(args)
# image, out_boxes = yolo_model.detect_image(path)
# print(out_boxes)
