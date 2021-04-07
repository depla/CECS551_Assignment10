import os

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

pics = os.scandir("img_celeba")
for entry in pics:
    print(entry)




