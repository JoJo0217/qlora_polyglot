SFT(supervised finetuning) 과정에서 사용되는 polyglot model을 QLoRA 방식으로 훈련하기 위한 코드입니다.
qlora.py를 통해 모델을 훈련시키면 lora 모델이 나오고 lora_merge.py을 통해 기존 base 모델에 합쳐서 저장을 할 수 있습니다.
