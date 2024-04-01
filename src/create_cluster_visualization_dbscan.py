import json
import numpy as np
from tqdm import tqdm

import umap
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go
from collections import Counter, defaultdict

INPUT_PREFIX   = "./sharegpt_data/sharegpt-en"

INPUT_TEXT     = f"{INPUT_PREFIX}.text.json"
INPUT_EMBEDS   = f"{INPUT_PREFIX}.lora8max_uembeds.jsonl"
OUTPUT_LABELS  = f"{INPUT_PREFIX}.lora8max_dbscan.txt"
OUTPUT_VISUAL  = f"{INPUT_PREFIX}.lora8max_dvisual.jsonl"

# Load dataset

with open(INPUT_TEXT, "r") as f_t, open(INPUT_EMBEDS, "r", encoding="utf-8") as f_e:
    texts = json.load(f_t)
    texts = [t.replace("\n\nAssistant:", "<|end_of_turn|>Assistant:").replace("\n\nHuman:", "<|end_of_turn|>Human:") for t in texts]
    embeddings = [e.strip() for e in f_e.readlines()]
    embeddings = np.array([json.loads(e) for e in embeddings])
    assert len(texts) == len(embeddings)

    # Assume all embeddings have unit length
    # Then Euclidean distance ||x - y|| = 2(1 - cos(x, y))
    # assert np.isclose(np.linalg.norm(embeddings, axis=-1), 1.0).all()

print("DBSCAN clustering from scratch...")
km = DBSCAN(eps=1.0, min_samples=10).fit(embeddings)
labels = km.labels_

labs = sorted(list(set(labels)))
counter = Counter(labels)
print("Totally "+str(len(labs))+" labels: "+str([v for k,v in counter.most_common()]))
print(str(counter[-1])+" are labeled as -1.")

if -1 in [k for k,v in counter.most_common()[:10]]:
    common = [k for k,v in counter.most_common()[:10]]
    common.remove(-1)
else:
    common = [k for k,v in counter.most_common()[:9]]
    
mappin = defaultdict(lambda: 10)
for i, l in enumerate(common):
    mappin[l] = i

with open(OUTPUT_LABELS, "w") as f_k:
    for l in list(labels):
        f_k.write(str(mappin[l])+"\n")

with open(OUTPUT_VISUAL, "w") as f:
    for line in zip(texts, embeddings.tolist(), labels.tolist()):
        f.write(json.dumps({"text": line[0], "embedding": line[1], "color": mappin[line[2]]}, ensure_ascii=False)+"\n")