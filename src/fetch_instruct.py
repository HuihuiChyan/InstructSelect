import json
with open("humanmix.data.jsonl", "r") as fin,\
open("humanmix.inst.jsonl", "w") as fout:
    lines = [json.loads(line.strip()) for line in fin.readlines()]
    for line in lines:
        line["messages"] = line["messages"][:1]
        fout.write(json.dumps(line, ensure_ascii=False)+"\n")