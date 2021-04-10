from imageFinder import imageFinder, save_embeddings_into_json
import numpy as np
import random
import matplotlib.pyplot as plt

# list_of_taus = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
# list_of_taus = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1,
#                 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0]
# list_of_taus = [0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6]
interval_taus = 0.05
num_decimals = len(str(interval_taus).split('.')[1])
max_range = 2 + interval_taus
list_of_taus = np.arange(0, max_range, interval_taus)
list_of_taus = np.around(list_of_taus, num_decimals)

# run to create json file of all embeddings
# save_embeddings_into_json('selected_jpgs')

# get dict of jpg-celebs
identity_celebA_file = open("identity_CelebA.txt", "r")
jpg_celebs_answers_dict = {}

for line in identity_celebA_file:
    line = line.strip()
    jpg, celeb = line.split(" ")
    jpg_celebs_answers_dict[jpg] = celeb

identity_celebA_file.close()

# get random 10 celebs
selected_id_file = open("selected_ids.txt", "r")
selected_ids = []
for id in selected_id_file:
    selected_ids.append(id.strip())

selected_id_file.close()

list_of_celebs = random.sample(selected_ids, 10)
print("List of celebs:", list_of_celebs)

# find anchor pics
anchors_dict = {}
for person in list_of_celebs:
    for jpg in jpg_celebs_answers_dict:
        if person == jpg_celebs_answers_dict[jpg]:
            anchors_dict[jpg] = jpg_celebs_answers_dict[jpg]
            break

print("Celeb Anchors Dict:", anchors_dict)

# get data for plots
# each celeb id will have dict of taus and each tau has precision and recall
celeb_metrics = {}
num_actual_images_of_celeb = 29
celeb_counter = 0
for anchor_pic in anchors_dict:
    celeb_counter += 1
    taus_for_celeb_dict = {}

    for tau in list_of_taus:
        print("Working on celeb", celeb_counter, "out of", len(anchors_dict), "Current Tau:", tau)
        num_correctly_recognized = 0

        # find a list of images that we think are similar to the anchor
        found_images = imageFinder(anchor_pic, tau)
        num_recognized = len(found_images)

        if len(found_images) > 0:
            # count how many predictions we got correct
            # get the true id of the anchor image
            true_id = jpg_celebs_answers_dict[anchor_pic]
            for found_image in found_images:
                # get the id of the found image
                found_id = jpg_celebs_answers_dict[found_image]

                if true_id == found_id:
                    num_correctly_recognized += 1

        # calculate the precision and recall
        if num_recognized == 0:
            # div by 0 case
            precision = 1
        else:
            precision = num_correctly_recognized / num_recognized

        recall = num_correctly_recognized / num_actual_images_of_celeb

        metrics_dict = {'precision': precision, 'recall': recall}
        taus_for_celeb_dict[tau] = metrics_dict

    celeb_metrics[anchors_dict[anchor_pic]] = taus_for_celeb_dict

print(celeb_metrics)

# plot precision
for celeb_id in celeb_metrics:
    x_data = []
    y_data = []
    tau_dict = celeb_metrics[celeb_id]
    for tau in tau_dict:
        metrics = tau_dict[tau]
        precision = metrics['precision']
        x_data.append(tau)
        y_data.append(precision)

    label = "ID: " + celeb_id
    plt.plot(x_data, y_data, label=label)

# show precision plot
plt.title('Precision Plot')
plt.xlabel('Tau')
plt.ylabel('Precision')
# plt.xticks(np.arange(0, 2.1, 0.1))
plt.legend()
plt.show()



# plot recall
for celeb_id in celeb_metrics:
    x_data = []
    y_data = []
    tau_dict = celeb_metrics[celeb_id]
    for tau in tau_dict:
        metrics = tau_dict[tau]
        recall = metrics['recall']
        x_data.append(tau)
        y_data.append(recall)

    label = "ID: " + celeb_id
    plt.plot(x_data, y_data, label=label)

# show recall plot
plt.title('Recall Plot')
plt.xlabel('Tau')
plt.ylabel('Recall')
# plt.xticks(np.arange(0, 2.1, 0.1))
plt.legend()
plt.show()
