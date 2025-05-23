#!/bin/bash
CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS=ignore python main.py --resume checkpoint.pth.tar --batch-size 128 --n-conv 5 --n-h 1 --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 --workers 10 --epochs 300 --print-freq 1 data/relaxed_structures_hse_bg >> cgcnn_train.out
