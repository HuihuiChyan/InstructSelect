import numpy as np
import json
import random

random.seed(42)
np.random.seed(42)

SAMPLE_SIZE   = 2000

INPUT_PREFIX  = "./sharegpt_data/sharegpt"
OUTPUT_PREFIX = "./sharegpt_data/sharegpt-random2k"

INPUT_DATA    = f"{INPUT_PREFIX}.data.jsonl"

OUTPUT_DATA   = f"{OUTPUT_PREFIX}.data.jsonl"

with open(INPUT_DATA, "r") as fd_in:

    data = [json.loads(d.strip()) for d in fd_in.readlines()]

with open(OUTPUT_DATA, "w") as fout_d:
    indices = list(range(len(data)))
    selected_indices = random.sample(indices, k=SAMPLE_SIZE)
    for idx in selected_indices:
        fout_d.write(json.dumps(data[idx], ensure_ascii=False)+"\n")