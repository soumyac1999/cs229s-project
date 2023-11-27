import os 
import torch 
import matplotlib.pyplot as plt

ckpt_dir = 'wikitext'
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
plt.savefig('fig')
plt.clf()
