
# Tensor Model Parallelism (Part 1b)

## Run the following commands

To do a training run using tensor model parallelism with a batch size of 4, run:
```bash
python train.py config/train_wikitext.py \
    --max_iters=40 --batch_size=4 --block_size=128 \
    --gradient_accumulation_steps=40 --init_from=scratch \
    --eval_iters=1
```

To reproduce the plots for tensor model parallelism, run:
```bash
python plot_runs.py
```
