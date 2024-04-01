import numpy as np
import json
import random

random.seed(42)
np.random.seed(42)

SAMPLE_SIZE   = 10000

INPUT_PREFIX  = "./sharegpt_data/sharegpt-lora8max100k"
OUTPUT_PREFIX = "./sharegpt_data/sharegpt-lora8max100k-maxmix10k"

INPUT_DATA    = f"{INPUT_PREFIX}.data.jsonl"
INPUT_ATTR1   = f"{INPUT_PREFIX}.reward.txt"
INPUT_ATTR2   = f"{INPUT_PREFIX}.uct.txt"

OUTPUT_DATA   = f"{OUTPUT_PREFIX}.data.jsonl"

def normalize(x):
    x = (x-x.min())/(x.max()-x.min())

    return x

with open(INPUT_DATA, "r") as fd_in:
    data = [json.loads(d.strip()) for d in fd_in.readlines()]
    
with open(INPUT_ATTR1, "r") as fin:
    attrs1 = np.array([float(line.strip()) for line in fin.readlines()])
    attrs1 = normalize(attrs1)

with open(INPUT_ATTR2, "r") as fin:
    # attrs2 = json.load(fin)
    # attrs2 = np.array(attrs2['mc_means'])/np.array(attrs2['mc_vars'])
    attrs2 = np.array([float(line.strip()) for line in fin.readlines()])
    attrs2 = normalize(attrs2)

assert len(data) == len(attrs1) == len(attrs2)

with open(OUTPUT_DATA, "w") as fout_d:
    attrs = attrs1 + attrs2
    # attrs = attrs1
    indices = attrs.argsort()[-SAMPLE_SIZE:]
    for idx in indices:
        fout_d.write(json.dumps(data[idx], ensure_ascii=False)+"\n")