from subprocess import Popen, PIPE
import matplotlib.pyplot as plt


def plot(results):
    batch_sizes = list(results.keys())
    throughput_values = [result[0] for result in results.values()]
    memory_usage_strings = [result[1] for result in results.values()]

    # Convert memory usage strings to lists of floats
    memory_usage_values = [list(map(float, mem.split(', '))) for mem in memory_usage_strings]

    # Plot Training Throughput
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(batch_sizes, throughput_values, marker='o')
    plt.title('Training Throughput vs Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Throughput')

    # Plot Memory Usage on 4 GPUs
    plt.subplot(1, 2, 2)
    for i in range(4):
        plt.plot(batch_sizes, [mem[i] for mem in memory_usage_values], label=f'GPU {i + 1}', marker='o')

    plt.title('Memory Usage on 4 GPUs vs Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Memory Usage')
    plt.legend()

    plt.tight_layout()
    plt.savefig('tp_plot.pdf', dpi=300)


results = {}

bs = 4
while True:
    config = (f"--max_iters=40 --batch_size={bs} --block_size=128 "
              "--gradient_accumulation_steps=40 --init_from=scratch "
              "--eval_iters=1")
    out = Popen(
        f"python train.py config/train_wikitext.py {config}", 
        shell=True, stdout=PIPE, text=True).stdout.read()
    sentinel, _, training_throughput, _, mem_usage = out.decode(
        "utf-8").split("\n")[-2].split(" ")
    assert sentinel == "SENTINEL"
    print(bs, training_throughput, mem_usage)
    results[bs] = (training_throughput, mem_usage)
    plot(results)

    bs *= 2
