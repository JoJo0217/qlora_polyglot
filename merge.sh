#!/bin/bash
python lora_merge.py \
    --base-model-path EleutherAI/polyglot-ko-12.8b \
    --target-model-path /dse/qlora/output \
    --lora-path /dse/qlora/outputs
    

