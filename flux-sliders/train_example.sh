#!/bin/bash

python3 train_flux_sliders.py \
    --model_path "/workspace/models/FLUX.1-Krea-dev" \
    --output_dir "/workspace/models/fluxsliders" \
    --target_prompt "photo of a person" \
    --positive_prompt "photo of a person, extremely tall person, very tall, towering height, long legs, tall stature, small head, elongated proportions, long limbs, tiny head" \
    --negative_prompt "photo of a person, very short person, extremely short, small stature, petite, tiny, short legs, low height, large head, big head, compact proportions" \
    --slider_name "person-height" \
    --max_train_steps 1000 \
    --learning_rate 0.002 \
    --rank 16 \
    --alpha 1 \
    --eta 2.0 \
    --height 512 \
    --width 512 \
    --save_every 300
