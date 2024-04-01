import json
import random
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
INPUT_EMBEDS = "./sharegpt_data/sharegpt.lora8max_uembeds.jsonl"
with open(INPUT_EMBEDS) as fin:
    lines = [json.loads(line.strip()) for line in fin.readlines()]
    random.shuffle(lines)
    lines = lines[:10000]
    similarity_matrix = euclidean_distances(lines, lines)
    avg_similar = similarity_matrix.mean()
    similarity_matrix = similarity_matrix + np.eye(len(lines)) * 99999999
    min_similar = similarity_matrix.min(axis=0).mean()
    print(f"minimal density is {min_similar}.")
    print(f"average density is {avg_similar}.")