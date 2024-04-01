import json
import numpy as np
INPUT = "./sharegpt_data/sharegpt.unnorm_lora8max_embeds.jsonl"
OUTPUT = "./sharegpt_data/sharegpt.lora8max_embeds.jsonl"
with open(INPUT, "r") as fin,\
open(OUTPUT, "w") as fout:
    lines = np.array([json.loads(line.strip()) for line in fin.readlines()])
    lines = lines/np.expand_dims(np.linalg.norm(lines, axis=-1), axis=-1)
    for line in lines:
        fout.write(json.dumps(line.tolist())+"\n")