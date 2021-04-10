from keras.models import load_model
import os
from PIL import Image
import numpy as np
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot
from sklearn.preprocessing import normalize
import argparse
import sys
import os

from yoloface.utils import *

"""
FaceNet Model from:
https://github.com/nyoki-mtl/keras-facenet
Code from:
https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/
using MTCNN library
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
downscale_size = (160, 160)

print("Loading Yolo v3 Model")
net = cv2.dnn.readNetFromDarknet(args.model_cfg, args.model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# extract a single face from a given photograph
def image2vect(filename):
    # load image from file
    # image = Image.open(filename)
    # # convert to array
    # pixels = np.asarray(image)

    cap = cv2.VideoCapture(filename)
    has_frame, frame = cap.read()
    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                 [0, 0, 0], 1, crop=False)
    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(get_outputs_names(net))

    # Remove the bounding boxes with low confidence
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
    # print(type(image))
    face = np.asarray(image)
    image = Image.fromarray(face)
    image = image.resize(downscale_size)
    face_array = np.asarray(image)
    # pyplot.imshow(face_array)
    # pyplot.show()
    # print(face_array.shape)

    # # create the detector, using default weights
    # detector = MTCNN()
    # # detect faces in the image
    # results = detector.detect_faces(pixels)
    # # check if a face was found
    # if len(results) == 0:
    #     # did not find face
    #     print("Cannot detect face for file!", filename)
    #     # return an embedding of all 100
    #     return np.full((1, 128), 100)
    # # extract the bounding box from the first face
    # x1, y1, width, height = results[0]['box']
    # x1, y1 = abs(x1), abs(y1)
    # x2, y2 = x1 + width, y1 + height
    # # extract the face
    # face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size


    # image = Image.fromarray(face)
    # image = image.resize(downscale_size)
    # face_array = np.asarray(image)
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
#
# sum = 0
# for i in range(len(embedding[0])):
#     sum += pow(embedding[0][i], 2)
# print(sum)


# import argparse
# import sys
# import os
#
# from yoloface.utils import *

#####################################################################
# parser = argparse.ArgumentParser()
# parser.add_argument('--model-cfg', type=str, default='./yoloface/cfg/yolov3-face.cfg',
#                     help='path to config file')
# parser.add_argument('--model-weights', type=str,
#                     default='./yoloface/model-weights/yolov3-wider_16000.weights',
#                     help='path to weights of model')
# parser.add_argument('--image', type=str, default='selected_jpgs/004659.jpg',
#                     help='path to image file')
# parser.add_argument('--video', type=str, default='',
#                     help='path to video file')
# parser.add_argument('--src', type=int, default=0,
#                     help='source of the camera')
# parser.add_argument('--output-dir', type=str, default='outputs/',
#                     help='path to the output directory')
# args = parser.parse_args()

#####################################################################
# print the arguments
# print('----- info -----')
# print('[i] The config file: ', args.model_cfg)
# print('[i] The weights of model file: ', args.model_weights)
# print('[i] Path to image file: ', args.image)
# print('[i] Path to video file: ', args.video)
# print('###########################################################\n')

# check outputs directory
# if not os.path.exists(args.output_dir):
#     print('==> Creating the {} directory...'.format(args.output_dir))
#     os.makedirs(args.output_dir)
# else:
#     print('==> Skipping create the {} directory...'.format(args.output_dir))

# Give the configuration and weight files for the model and load the network
# using them.
# net = cv2.dnn.readNetFromDarknet(args.model_cfg, args.model_weights)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# def _main():
#     wind_name = 'face detection using YOLOv3'
#     cv2.namedWindow(wind_name, cv2.WINDOW_NORMAL)
#
#     output_file = ''
#
#     if args.image:
#         if not os.path.isfile(args.image):
#             print("[!] ==> Input image file {} doesn't exist".format(args.image))
#             sys.exit(1)
#         cap = cv2.VideoCapture(args.image)
#         output_file = args.image[:-4].rsplit('/')[-1] + '_yoloface.jpg'

    # while True:
    #
    #     has_frame, frame = cap.read()

        # Stop the program if reached end of video
        # if not has_frame:
        #     print('[i] ==> Done processing!!!')
        #     print('[i] ==> Output file is stored at', os.path.join(args.output_dir, output_file))
        #     cv2.waitKey(1000)
        #     break

        # Create a 4D blob from a frame.
        # blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
        #                              [0, 0, 0], 1, crop=False)

        # # Sets the input to the network
        # net.setInput(blob)
        #
        # # Runs the forward pass to get output of the output layers
        # outs = net.forward(get_outputs_names(net))
        #
        # # Remove the bounding boxes with low confidence
        # faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)

        # left top right bottom
#         print(faces)
#         box_data = faces[0]
#         left, top, right, bottom = refined_box(box_data[0], box_data[1], box_data[2], box_data[3])
#         image = Image.open(args.image)
#         image = image.crop((left, top, right, bottom))
#         print(type(image))
#         face = np.asarray(image)
#         image = Image.fromarray(face)
#         image = image.resize(downscale_size)
#         face_array = np.asarray(image)
#         pyplot.imshow(face_array)
#         pyplot.show()
#         print(face_array.shape)
#
#
#         print('[i] ==> # detected faces: {}'.format(len(faces)))
#         print('#' * 60)
#
#         # initialize the set of information we'll displaying on the frame
#         info = [
#             ('number of faces detected', '{}'.format(len(faces)))
#         ]
#
#         for (i, (txt, val)) in enumerate(info):
#             text = '{}: {}'.format(txt, val)
#             cv2.putText(frame, text, (10, (i * 20) + 20),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)
#
#         # Save the output video to file
#         cv2.imwrite(os.path.join(args.output_dir, output_file), frame.astype(np.uint8))
#
#
#         cv2.imshow(wind_name, frame)
#
#         key = cv2.waitKey(1)
#         if key == 27 or key == ord('q'):
#             print('[i] ==> Interrupted by user!')
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
#     print('==> All done!')
#     print('***********************************************************')
#
#
# if __name__ == '__main__':
#     _main()

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
