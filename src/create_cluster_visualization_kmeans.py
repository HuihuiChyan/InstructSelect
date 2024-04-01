import os
import json
import numpy as np
from tqdm import tqdm

import umap
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go

INPUT_TEXT     = "./human_mix_data/humanmix-1w.text.json"
INPUT_EMBEDS   = "./human_mix_data/humanmix-1w.lora8max-embeds.jsonl"
UMAP_EMBEDS    = "./human_mix_data/humanmix-1w.lora8max-uembeds.jsonl"
OUTPUT_KMEANS  = "./human_mix_data/humanmix-1w.lora8max_kmeans.jsonl"
OUTPUT_VISUAL  = "./human_mix_data/humanmix-1w.lora8max_kvisual.jsonl"

# Load dataset

with open(INPUT_TEXT, "r") as f_t, open(INPUT_EMBEDS, "r", encoding="utf-8") as f_e:
    texts = json.load(f_t)
    texts = [t.replace("\n\nAssistant:", "<|end_of_turn|>Assistant:").replace("\n\nHuman:", "<|end_of_turn|>Human:") for t in texts]
    embeddings = [e.strip() for e in f_e.readlines()]
    embeddings = np.array([json.loads(e) for e in embeddings])
    assert len(texts) == len(embeddings)

    # Assume all embeddings have unit length
    # Then Euclidean distance ||x - y|| = 2(1 - cos(x, y))
    assert np.isclose(np.linalg.norm(embeddings, axis=-1), 1.0).all()

if not os.path.exists(UMAP_EMBEDS):
    print("Dimention reduction from scratch...")

    # Dimensional reduction
    transform = umap.UMAP(n_neighbors=500, min_dist=0.1, n_components=3, metric="euclidean", low_memory=False)
    embeddings_umap = transform.fit_transform(embeddings)
    with open(UMAP_EMBEDS, "w") as f_u:
        for embed in embeddings_umap.tolist():
            f_u.write(json.dumps(embed)+"\n")
else:
    with open(UMAP_EMBEDS, "r") as f_u:
        embeddings_umap = np.array([json.loads(line.strip()) for line in f_u.readlines()])

if not os.path.exists(OUTPUT_KMEANS):
    print("K-means clustering from scratch...")

    # K-Means classification
    scores = []
    labels = []
    for k_cluster in [20]:
        km = KMeans(k_cluster, n_init=10).fit(embeddings_umap)
        labels.append(km.labels_)
        scores.append(silhouette_score(embeddings_umap, km.labels_, metric="euclidean"))

    # Use optimal K as pseudo labels
    print("The silhouette scores are: "+str(scores))
    labels = labels[np.argmax(scores)]
    with open(OUTPUT_KMEANS, "w") as f_k:
        json.dump(labels.tolist(), f_k)

else:
    print("Loading pre-computed K-means clusters...")
    with open(OUTPUT_KMEANS, "r") as f_k:
        labels = np.array(json.load(f_k))
        assert len(texts) == len(labels)

# 3D Figure
fig = go.Figure(data=[
    go.Scatter3d(
        x=embeddings_umap[:, 0], y=embeddings_umap[:, 1], z=embeddings_umap[:, 2],
        mode="markers",
        marker=dict(
            size=1.5,
            opacity=0.8,
            color=labels
        )
    )
])

fig.show()

with open(OUTPUT_VISUAL, "w") as f:
    for line in zip(texts, embeddings_umap.tolist(), labels.tolist()):
        f.write(json.dumps({"text": line[0], "embedding": line[1], "color": line[2]}, ensure_ascii=False)+"\n")