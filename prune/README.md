
# Pruning (Part 2)
## Run the following commands

### L1 Unstructured, Part A
```bash
$ python train.py \
    config/train_wikitext.py \
    --max_iters=100 \
    --lr_decay_iters=100 \ 
    --prune_l1=True \
    --prune_max_iters=30
```
### L1 Unstructured, Part B
```bash
$ python train.py \
    config/train_wikitext.py \
    --max_iters=50 \
    --lr_decay_iters=50 \ 
    --prune_l1=True \
    --prune_max_iters=60 \
    --loss_guided_prune=True
```
### L2 Structured, Part A
```bash
$ python train.py \
    config/train_wikitext.py \
    --max_iters=100 \
    --lr_decay_iters=100 \ 
    --prune_l2=True \
    --prune_max_iters=30 \
    --loss_guided_prune=True
```
### L1 Unstructured, Part B
```bash
$ python train.py \
    config/train_wikitext.py \
    --max_iters=50 \
    --lr_decay_iters=50 \ 
    --prune_l2=True \
    --prune_max_iters=60 \
    --loss_guided_prune=True
```