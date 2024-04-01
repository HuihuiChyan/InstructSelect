import json
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances

random.seed(42)
np.random.seed(42)

INPUT_PREFIX  = "./sharegpt_data/sharegpt"
OUTPUT_PREFIX = "./sharegpt_data/sharegpt-en"

INPUT_TEXT    = f"{INPUT_PREFIX}.text.json"
INPUT_DATA    = f"{INPUT_PREFIX}.data.jsonl"
INPUT_LABELS  = f"{INPUT_PREFIX}.lora8max_dbscan.txt"
INPUT_EMBEDS  = f"{INPUT_PREFIX}.lora8max_uembeds.jsonl"

OUTPUT_TEXT   = f"{OUTPUT_PREFIX}.text.json"
OUTPUT_DATA   = f"{OUTPUT_PREFIX}.data.jsonl"
OUTPUT_EMBEDS = f"{OUTPUT_PREFIX}.lora8max_uembeds.jsonl"

with open(INPUT_TEXT, "r") as ft_in, open(INPUT_DATA, "r") as fd_in, \
open(INPUT_EMBEDS, "r", encoding="utf-8") as fe_in, open(INPUT_LABELS, "r") as fl_in:

    texts = json.load(ft_in)
    data = [json.loads(d.strip()) for d in fd_in.readlines()]
    embeddings = [e.strip() for e in fe_in.readlines()]
    embeddings = np.array([json.loads(e) for e in embeddings])
    labels = [line.strip() for line in fl_in.readlines()]

    assert len(texts) == len(data) == len(embeddings) == len(labels)
    
text_output = []
with open(OUTPUT_TEXT, "w") as ft_out, open(OUTPUT_EMBEDS, "w", encoding="utf-8") as fe_out,\
open(OUTPUT_DATA, "w", encoding="utf-8") as fd_out:
    for idx, lab in enumerate(labels):
        if lab == '0' or lab == '1':
            text_output.append(texts[idx])
            fe_out.write(json.dumps(embeddings[idx].tolist())+"\n")
            fd_out.write(json.dumps(data[idx], ensure_ascii=False)+"\n")
                
    json.dump(text_output, ft_out, ensure_ascii=False, indent=4)