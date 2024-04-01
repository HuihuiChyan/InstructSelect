import json
import random
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
INPUT_EMBEDS = "./sharegpt_data/sharegpt.reward.txt"
with open(INPUT_EMBEDS, "r", encoding="utf-8") as fin:
    lines = [float(line.strip()) for line in fin.readlines()]
    avg_similar = np.array(lines).mean()
    print(f"average reward is {avg_similar}.")