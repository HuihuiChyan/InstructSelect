import json
import random
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
INPUT_EMBEDS = "./wizardlm_data/wizardlm.uct.json"
with open(INPUT_EMBEDS, "r", encoding="utf-8") as fin:
    lines = json.load(fin)
    ucts = []
    for mc_mean, mc_var in zip(lines['mc_means'], lines['mc_vars']):
        if mc_mean != 0:
            ucts.append(mc_var/mc_mean)
        else:
            ucts.append(0.0)
    avg_ucts = np.array(ucts).mean()
    print(f"average uct is {avg_ucts}.")