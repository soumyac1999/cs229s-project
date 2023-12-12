import matplotlib.pyplot as plt

# Given function
def plot(results):
    batch_sizes = list(results.keys())
    throughput_values = [float(result[0]) for result in results.values()]
    memory_usage_strings = [result[1] for result in results.values()]

    # Convert memory usage strings to lists of floats
    memory_usage_values = [list(map(float, mem.split(','))) for mem in memory_usage_strings]

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


# Values to feed into the function
results_dict = {
    4: ['1184.1638932804074', '2.158878208,0.0,0.0,0.0'],
    8: ['2382.961254541504', '2.518595072,0.0,0.0,0.0'],
    16: ['4612.84249293933', '3.2431488,0.0,0.0,0.0'],
    32: ['8538.482316600135', '4.695229952,0.0,0.0,0.0'],
    64: ['13692.174117921739', '7.591845376,0.0,0.0,0.0']
}

# Call the plot function
plot(results_dict)

