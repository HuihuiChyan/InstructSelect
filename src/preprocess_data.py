import os
import json
import random
import argparse
import transformers
from tqdm import tqdm
def generate_split(conversations, tokenizer, out_file, max_context=2048):

    role_prefix = {"user": "Human: ", "assistant": "Assistant: "}

    new_convs = []
    for conv in tqdm(conversations):
        tokens = "Assistant: "+conv["dataset"] + "\n\n"
        length = len(tokenizer.tokenize(tokens))

        # Messages
        for idx, message in enumerate(conv["messages"]):
            
            # Message
            curr_tokens = role_prefix[message["role"]]
            curr_tokens = curr_tokens + message["content"] + "\n\n"

            curr_length = len(tokenizer.tokenize(curr_tokens))

            if not length + curr_length > max_context:
                tokens += curr_tokens
                length += curr_length
            else:
                break
        
        new_convs.append(tokens)

    with open(out_file, "w") as f:
        json.dump(new_convs, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-path", type=str, default="./llama_7B")

    parser.add_argument("--in-file", type=str, default="./humanmix_data/humanmix.data.jsonl")
    parser.add_argument("--out-file", type=str, default="./humanmix_data/humanmix.inst.json")

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.in_file, "r") as f:
        conversations = [json.loads(line.strip()) for line in f.readlines()]

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_path)

    generate_split(conversations, tokenizer, out_file=args.out_file)