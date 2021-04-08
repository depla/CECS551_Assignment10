from keras.models import load_model
from yolo import yolo
import os


class Args:
    def __init__(self):
        self.model = 'YOLO_Face.h5'
        self.classes = 'cfg/face_classes.txt'
        self.anchors = 'cfg/yolo_anchors.txt'
        self.img_size = (416, 416)
        self.score = 0.5
        self.iou = 0.5


path = 'selected_jpgs/000109.jpg'

# load the facenet model
face_net_model = load_model('facenet_keras.h5')
# summarize input and output shape
print(face_net_model.inputs)
print(face_net_model.outputs)

args = Args()
yolo_model = yolo.YOLO(args)
image, out_boxes = yolo_model.detect_image(path)
print(out_boxes)

