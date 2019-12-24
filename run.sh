#!/bin/sh

export CUDA_VISIBLE_DEVICES="0"

BATCH_SIZE=8
WORKER_SIZE=2
MAX_EPOCHS=200

python ./main.py --batch_size $BATCH_SIZE --workers $WORKER_SIZE --use_attention --max_epochs $MAX_EPOCHS --dropout 0.3 --input_dropout 0.1 --max_len 120
