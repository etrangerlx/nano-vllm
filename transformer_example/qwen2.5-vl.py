from transformers import AutoProcessor, AutoModelForMultimodalLM

processor = AutoProcessor.from_pretrained("/workspace/models/Qwen2.5-VL-3B-Instruct")
model = AutoModelForMultimodalLM.from_pretrained("/workspace/models/Qwen2.5-VL-3B-Instruct", device_map="auto")
messages_vl = {
        "role": "user",
        "content": [
            {"type": "image", "url": "/workspace/project/nano-vllm/resource/images/cat.jpg"},
            {"type": "text", "text": "描述一下这个图片?"}
        ]
    }
messages_text = {
		"role": "user",
		"content": [
			{"type": "text", "text": "介绍一下自己"}
		]
	}


inputs = processor.apply_chat_template(
	[messages_vl, messages_text],
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512)
for output in outputs:
    print(processor.decode(output[inputs["input_ids"].shape[-1]:]))
