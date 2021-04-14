import os
import shutil
"""
CECS 551 - Assignment 10 - Yolo FaceNet
Alex Nassif
Dennis La
Sophanna Ek
"""

"""
Script to get the selected jpgs
"""
# get the selected ids
selected_id_file = open("selected_ids.txt", "r")
selected_ids = []
for id in selected_id_file:
    selected_ids.append(id.strip())

selected_id_file.close()

# get list of jpg-celebs
identity_celebA_file = open("identity_CelebA.txt", "r")
jpg_celebs = []

for line in identity_celebA_file:
    line = line.strip()
    jpg_celebs.append(line)

identity_celebA_file.close()

# get dict of selected jpg/celeb id pairs
selected_jpg_celeb_dict = {}

for id in selected_ids:
    for element in jpg_celebs:
        jpg, celeb_id = element.split(" ")
        if id == celeb_id:
            selected_jpg_celeb_dict[jpg] = celeb_id

print(len(selected_jpg_celeb_dict), "images selected")

pics = os.listdir("img_celeba")
for jpg in selected_jpg_celeb_dict:
    path = "img_celeba/" + jpg

    shutil.copy(path, 'selected_jpgs')

print(len(os.listdir("selected_jpgs")), "len of selected jpgs")





