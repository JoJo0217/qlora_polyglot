import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model

lora_r=128
lora_alpha=1
target_modules=["query_key_value"]
lora_dropout=0
model_id = "EleutherAI/polyglot-ko-12.8b"
data_path="/dse/qlora/data/train.json"
per_device_train_batch_size=8
gradient_accumulation_steps=1
warmup_steps=0 #inst gpt paper
train_epochs=2
learning_rate=9.65e-6 #inst gpt paper
fp16=True
logging_steps=50
log_dir='./logs'
save_strategy="steps"
save_steps=200
output_dir="./outputs"
optim="paged_adamw_8bit"
max_length=1024
val_set_size=0

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})
model.config.end_token_id = tokenizer.eos_token_id
model.config.pad_token_id = model.config.eos_token_id
model.resize_token_embeddings(int(8 *math.ceil(len(tokenizer) / 8.0)))

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=target_modules,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)
from datasets import load_dataset

data = load_dataset('json',data_files=data_path)
data=data.map(lambda x:{'text': x['prompt']+x['chosen']+'<|endoftext|>'})
data = data.map(lambda samples: tokenizer(samples["text"],max_length=max_length,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt"), batched=True)

if val_set_size > 0:
    train_val = data["train"].train_test_split(test_size=val_set_size, shuffle=True, seed=42)
    train_data = train_val["train"].shuffle()
    val_data = train_val["test"].shuffle()
else:
    train_data = data["train"].shuffle()
    val_data = None


import transformers

# needed for gpt-neo-x tokenizer
tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    #eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps= gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        #max_steps=50,
        num_train_epochs=train_epochs,
        learning_rate=learning_rate,
        bf16=True,
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=5,
        output_dir=output_dir,
        optim=optim,
        report_to='tensorboard',
        resume_from_checkpoint=True
        #evaluation_strategy="steps" if val_set_size > 0 else "no",
        #eval_steps=10 if val_set_size > 0 else None,
        #load_best_model_at_end=True if val_set_size > 0 else False,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
#
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

