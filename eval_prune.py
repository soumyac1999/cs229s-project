import os 
import torch 
import math
import matplotlib.pyplot as plt

ckpt_dir = 'wikitext_L1_part_a'
device = 'cpu'

dirs = [os.path.join(ckpt_dir, d) for d in os.listdir(ckpt_dir)]
dirs.sort(reverse=True)
print(dirs)
x = []
y1 = []
for d in dirs:
    ckpt = torch.load(d, map_location=device)
    x.append(1 - ckpt["pct_orig"])
    y1.append(ckpt["best_val_loss"])

plt.plot(x, y1)

y = math.log(16.12)
plt.plot([0.26, 0.26, 0], [0, y, y], 'k-', lw=1,dashes=[2, 2])

y = math.log(17.42)
plt.plot([0.48, 0.48, 0], [0, y, y], 'k-', color='green', lw=1, dashes=[2, 2])

y = math.log(19.47)
plt.plot([0.67, 0.67, 0], [0, y, y], 'k-', color='red', lw=1,dashes=[2, 2])

plt.ylim(2.0, 4.0)

plt.xlabel("Pct. Pruned")
plt.ylabel("Loss")
plt.title("Unstructured L1 Pruning, Protocol A")

plt.savefig('Unstructured_Part_A')
plt.clf()
