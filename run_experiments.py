import json
from subprocess import Popen, PIPE

# Run experiments (loss)
config = ("--max_iters=500 --batch_size=8 --block_size=1024 "
          "--gradient_accumulation_steps=40 --eval_interval=500")
out = Popen(
    f"torchrun --nproc_per_node=4 train.py config/train_wikitext.py {config}", 
    shell=True, stdout=PIPE).stdout.read()
sentinel, loss, _, _, _ = out.decode("utf-8").split("\n")[-2].split(" ")
assert sentinel == "SENTINEL"

# Run experiments (inference batch size 1)
config = "--max_iters=0 --batch_size=1 --block_size=1024 --eval_only=True"
out = Popen(
    f"torchrun --nproc_per_node=4 train.py config/train_wikitext.py {config}", 
    shell=True, stdout=PIPE).stdout.read()
sentinel, _, _, inference_latency_1, _ = out.decode(
    "utf-8").split("\n")[-2].split(" ")
assert sentinel == "SENTINEL"

# Run experiments (inference batch size 12)
config = "--max_iters=0 --batch_size=12 --block_size=1024 --eval_only=True"
out = Popen(
    f"torchrun --nproc_per_node=4 train.py config/train_wikitext.py {config}", 
    shell=True, stdout=PIPE).stdout.read()
sentinel, _, _, inference_latency_12, _ = out.decode(
    "utf-8").split("\n")[-2].split(" ")
assert sentinel == "SENTINEL"

# Run experiments (training throughput 4)
config = ("--max_iters=40 --batch_size=4 --block_size=1024 "
          "--gradient_accumulation_steps=40")
out = Popen(
    f"torchrun --nproc_per_node=4 train.py config/train_wikitext.py {config}", 
    shell=True, stdout=PIPE).stdout.read()
sentinel, _, training_throughput_4, _, _ = out.decode(
    "utf-8").split("\n")[-2].split(" ")
assert sentinel == "SENTINEL"

# Run experiments (training throughput 12)
config = ("--max_iters=40 --batch_size=12 --block_size=1024 "
          "--gradient_accumulation_steps=40")
out = Popen(
    f"torchrun --nproc_per_node=4 train.py config/train_wikitext.py {config}", 
    shell=True, stdout=PIPE).stdout.read()
sentinel, _, training_throughput_12, _, _ = out.decode(
    "utf-8").split("\n")[-2].split(" ")
assert sentinel == "SENTINEL"

with open('results.json', 'w') as f:
    result = {
        "loss": loss,
        "inference_latency_1": inference_latency_1,
        "inference_latency_12": inference_latency_12,
        "training_throughput_4": training_throughput_4,
        "training_throughput_12": training_throughput_12
    }
    json.dump(result, f)
