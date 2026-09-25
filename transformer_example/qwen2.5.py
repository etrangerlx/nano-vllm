from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("/workspace/models/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("/workspace/models/Qwen2.5-0.5B-Instruct", device_map="auto")
messages = [{"role": "user", "content": "介绍一下你自己"}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True,
)
print(response)