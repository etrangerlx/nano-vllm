from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("/workspace/models/Qwen2-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("/workspace/models/Qwen2-0.5B-Instruct", device_map="auto")
messages = [
    {"role": "user", "content": "简单的介绍一下自己"},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))