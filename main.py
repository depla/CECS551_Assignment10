from imageFinder import imageFinder, save_embeddings_into_json
import json
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

list_of_taus = [0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6]

save_embeddings_into_json('selected_jpgs')

# json_file = open('embedding_dict.json')
# dict = json.load(json_file)
# print(type(dict))
# dist = euclidean_distances(np.asarray(dict['195377.jpg']), np.asarray(dict['195377.jpg']))
# print(dist)

