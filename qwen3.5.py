from transformers import AutoProcessor, AutoModelForMultimodalLM

processor = AutoProcessor.from_pretrained("/mnt/g/Models/Qwen3.5-0.8B")
model = AutoModelForMultimodalLM.from_pretrained("/mnt/g/Models/Qwen3.5-0.8B", device_map="auto")

# The vision encoder supports at most 2304 patches (vision_config.num_position_embeddings),
# i.e. 2304 * 16 * 16 = 589,824 px (~768x768). The default config leaves longest_edge at
# 16.7M px, so a 3024x4032 photo is passed through at full size -> ~47k patches -> OOM.
processor.image_processor.size["longest_edge"] = 589_824
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "/home/leixian/nano-vllm/image.png"},
            {"type": "text", "text": "描述一下这个图片?"}
        ]
    },
]
inputs = processor.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))