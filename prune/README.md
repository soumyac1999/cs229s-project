
# Pruning (Part 3)
## Run the following commands

### L1 Unstructured, Part A
```
$ python train.py \
    config/train_wikitext.py \
    --max_iters=100 \
    --lr_decay_iters=100 \ 
    --prune_l1=True \
    --prune_max_iters=30
```
### L1 Unstructured, Part B
```
$ python train.py \
    config/train_wikitext.py \
    --max_iters=50 \
    --lr_decay_iters=50 \ 
    --prune_l1=True \
    --prune_max_iters=60 \
    --loss_guided_prune=True
```
### L2 Structured, Part A
```
$ python train.py \
    config/train_wikitext.py \
    --max_iters=100 \
    --lr_decay_iters=100 \ 
    --prune_l2=True \
    --prune_max_iters=30 \
    --loss_guided_prune=True
```
### L1 Unstructured, Part B
```
$ python train.py \
    config/train_wikitext.py \
    --max_iters=50 \
    --lr_decay_iters=50 \ 
    --prune_l2=True \
    --prune_max_iters=60 \
    --loss_guided_prune=True
```