import json
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances

random.seed(42)
np.random.seed(42)

SAMPLE_SIZE   = 100000

INPUT_PREFIX  = "./sharegpt_data/sharegpt"
OUTPUT_PREFIX = "./sharegpt_data/sharegpt-lora8max100k"

INPUT_TEXT    = f"{INPUT_PREFIX}.text.json"
INPUT_DATA    = f"{INPUT_PREFIX}.data.jsonl"
INPUT_RWD     = f"{INPUT_PREFIX}.reward.txt"
INPUT_UCT     = f"{INPUT_PREFIX}.uct.json"
INPUT_LABELS  = f"{INPUT_PREFIX}.lora8max_dbscan.txt"
INPUT_EMBEDS  = f"{INPUT_PREFIX}.lora8max_uembeds.jsonl"

OUTPUT_TEXT   = f"{OUTPUT_PREFIX}.text.json"
OUTPUT_DATA   = f"{OUTPUT_PREFIX}.data.jsonl"
OUTPUT_RWD    = f"{OUTPUT_PREFIX}.reward.txt"
OUTPUT_UCT    = f"{OUTPUT_PREFIX}.uct.txt"
OUTPUT_EMBEDS = f"{OUTPUT_PREFIX}.lora8max_uembeds.jsonl"

with open(INPUT_TEXT, "r") as ft_in, open(INPUT_DATA, "r") as fd_in, \
open(INPUT_EMBEDS, "r", encoding="utf-8") as fe_in:

    texts = json.load(ft_in)
    data = [json.loads(d.strip()) for d in fd_in.readlines()]
    embeddings = [e.strip() for e in fe_in.readlines()]
    embeddings = np.array([json.loads(e) for e in embeddings])

    assert len(texts) == len(data) == len(embeddings)

with open(INPUT_RWD, "r") as frwd_in, open(INPUT_UCT, "r") as fuct_in:

    rwd_lines = np.array([float(line.strip()) for line in frwd_in.readlines()])
    uct_lines = json.load(fuct_in)
    uct_lines = np.array(uct_lines['mc_means'])/np.array(uct_lines['mc_vars'])

    assert len(texts) == len(rwd_lines) == len(uct_lines)

if INPUT_LABELS != None:
    with open(INPUT_LABELS, "r") as fl_in:
        labels = [line.strip() for line in fl_in.readlines()]
    
    assert len(texts) == len(labels)

embed_lines = embeddings
indices = np.arange(len(texts))
new_idx = 0
selected_indices = [0]
remained_indices = indices[1:]
minimumm_lengths = None
for i in tqdm(range(SAMPLE_SIZE)):
    selected_lengths = euclidean_distances(embed_lines[new_idx:new_idx+1], embed_lines[remained_indices]).squeeze()
    if minimumm_lengths is None:
        minimumm_lengths = selected_lengths
    else:
        minimumm_lengths = np.vstack([minimumm_lengths, selected_lengths])
        minimumm_lengths = minimumm_lengths.min(axis=0)
    
    selected_remained_index = minimumm_lengths.argmax()
    new_idx = remained_indices[selected_remained_index]
    selected_indices.append(new_idx)
    remained_indices = np.delete(remained_indices, selected_remained_index)
    minimumm_lengths = np.delete(minimumm_lengths, selected_remained_index)
    
text_output = []
with open(OUTPUT_TEXT, "w") as ft_out, open(OUTPUT_EMBEDS, "w", encoding="utf-8") as fe_out,\
open(OUTPUT_DATA, "w", encoding="utf-8") as fd_out:
    for idx in selected_indices:
        text_output.append(texts[idx])
        if INPUT_LABELS is not None:
            data[idx]["dataset"] = "sharegpt" + labels[idx]
        fe_out.write(json.dumps(embeddings[idx].tolist())+"\n")
        fd_out.write(json.dumps(data[idx], ensure_ascii=False)+"\n")
            
    json.dump(text_output, ft_out, ensure_ascii=False, indent=4)

if rwd_lines is not None and uct_lines is not None:
    with open(OUTPUT_UCT, "w") as fu_out, open(OUTPUT_RWD, "w") as fr_out:
        for idx in selected_indices:
            fu_out.write(str(rwd_lines[idx])+"\n")
            fr_out.write(str(uct_lines[idx])+"\n")