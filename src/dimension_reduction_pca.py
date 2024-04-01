import sys
import json
import umap
import time
import random
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA

INPUT_EMBEDS   = "./sharegpt_data/sharegpt.lora8max_embeds.jsonl"
UMAP_EMBEDS    = "./sharegpt_data/sharegpt.lora8max_uembeds.jsonl"

with open(INPUT_EMBEDS, "r", encoding="utf-8") as f_e:
    embeddings = [e.strip() for e in f_e.readlines()]
    embeddings = np.array([json.loads(e) for e in embeddings])

    # Assume all embeddings have unit length
    # Then Euclidean distance ||x - y|| = 2(1 - cos(x, y))
    # assert np.isclose(np.linalg.norm(embeddings, axis=-1), 1.0).all()

transform = PCA(n_components=32)
embeddings = transform.fit_transform(embeddings)

print("PCA Dimention reduction finished")

prev_time = time.time()
transform = umap.UMAP(n_neighbors=50, min_dist=0.2, n_components=3, metric="euclidean", low_memory=False)
transform.fit(embeddings)
curr_time = time.time()
print(f"Umap learning finished! {curr_time-prev_time} seconds!")
embeddings_umap = transform.transform(embeddings)
next_time = time.time()
print(f"Umap reduction finished! {next_time-curr_time} seconds!")


with open(UMAP_EMBEDS, "w") as f_u:
    for embed in embeddings_umap.tolist():
        f_u.write(json.dumps(embed)+"\n")

# 3D Figure
fig = go.Figure(data=[
    go.Scatter3d(
        x=embeddings_umap[:, 0], y=embeddings_umap[:, 1], z=embeddings_umap[:, 2],
        mode="markers",
        marker=dict(
            size=1.5,
            opacity=0.8,
        )
    )
])

fig.show()