import os
import numpy as np
from image2vect import image2vect
from sklearn.metrics.pairwise import euclidean_distances
from matplotlib import pyplot
from PIL import Image

folder_name = 'selected_jpgs/'


def imageFinder(input_image, tau):
    image = Image.open(folder_name + input_image)
    image_pix = np.asarray(image)
    pyplot.imshow(image_pix)
    pyplot.show()

    list_of_images_of_celeb = []
    # get the embedding for the input image
    input_embedding = image2vect(folder_name + input_image)
    counter = 0
    # open the directory
    pics = os.listdir(folder_name)
    for pic in pics:
        # skip the input pic
        if pic != input_image:
            counter += 1
            print("Working on distance", counter, "out of 1199. Pic name:", pic)
            # find the embedding for the other pic
            other_embedding = image2vect(folder_name + pic)
            distance = euclidean_distances(input_embedding, other_embedding)
            print(distance)
            # we found a similar picture
            if distance < tau:
                image_same = Image.open(folder_name + pic)
                image_same_pix = np.asarray(image_same)
                pyplot.imshow(image_same_pix)
                pyplot.show()
                list_of_images_of_celeb.append(pic)

    return list_of_images_of_celeb




# test
image = '054110.jpg'
list = imageFinder(image, 1)
print(len(list))
print(type(list))
print(list)
