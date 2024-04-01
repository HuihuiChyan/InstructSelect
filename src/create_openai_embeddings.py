"""
- Embed text documents using OpenAI embedding API.

Usage:
python -m ochat.visualization.openai_embedding --in-file dataset_processed/ochat_text.json --out-file dataset_processed/ochat_text_embeddings.json
"""


import tiktoken
import openai
import json
import argparse
import os
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm import tqdm


# Text preprocessing

TEXT_BOS = "<s>"
TEXT_PROMPT = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
TEXT_REPLACE_TABLE = {"<|end_of_turn|>": "\n\n"}

# API

MAX_TOKENS = 8191
BATCH_SIZE = 128

MODEL_TYPE = "text-embedding-ada-002"
MODEL_TOKENIZER = tiktoken.encoding_for_model(MODEL_TYPE)

############


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def embedding_with_backoff(**kwargs):
    return openai.Embedding.create(**kwargs)


def preprocess_text(text: str):
    # Preprocess text, remove bos, add prompt and replace
    if text.startswith(TEXT_BOS):
        text = text[len(TEXT_BOS):]

    # for src, dst in TEXT_REPLACE_TABLE.items():
    #     text = text.replace(src, dst)

    # text = TEXT_PROMPT + text

    text = text.split("\n\nAssistant:")[0].strip()

    assert len(text) != 0

    # Tokenize and truncate
    tokens = MODEL_TOKENIZER.encode(text, disallowed_special=())
    tokens = tokens[:MAX_TOKENS]
    return tokens


def calculate_embeddings(samples, fout):

    for start_idx in tqdm(range(0, len(samples), BATCH_SIZE)):
        # Obtain a chunk
        sample_chunk = samples[start_idx: start_idx + BATCH_SIZE]

        # To tokens
        tokens_chunk = list(map(preprocess_text, sample_chunk))

        # Call API
        # Reference: https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_Wikipedia_articles_for_search.ipynb

        response = embedding_with_backoff(model=MODEL_TYPE, input=tokens_chunk)
        for i, be in enumerate(response["data"]):
            assert i == be["index"]  # double check embeddings are in same order as input
        
        embeddings_chunk = [e["embedding"] for e in response["data"]]

        for embedding in embeddings_chunk:
            fout.write(json.dumps(embedding)+"\n")

def main(args):
    openai.api_key = args["api_key"]
    samples = json.load(open(args["in_file"], "r"))
    if args["continue_generate"] == "True" and os.path.exists(args["out_file"]):
        with open(args["out_file"], "r", encoding="utf-8") as fin:
            lines = [line.strip() for line in fin.readlines()]
        with open(args["out_file"], "a", encoding="utf-8") as fout:
            calculate_embeddings(samples[len(lines):], fout)
    else:
        with open(args["out_file"], "w", encoding="utf-8") as fout:
            calculate_embeddings(samples, fout)        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, default="./sharegpt_data/sharegpt.text.json")
    parser.add_argument("--out-file", type=str, default="./sharegpt_data/sharegpt.inst_embeds.jsonl")
    parser.add_argument("--api-key", type=str, default="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")
    parser.add_argument("--continue-generate", type=str, choices=("True", "False"), default="True")
    args = parser.parse_args()

    main(vars(args))