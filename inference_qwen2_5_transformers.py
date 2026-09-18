import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread


def main():
    # 模型路径
    model_path = os.path.expanduser("E:/models/Qwen2.5-0.5B")

    print(f"Loading tokenizer and model from: {model_path}")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print("Model loaded.")

    # 采样参数
    gen_config = {
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "repetition_penalty": 1.1,
        "do_sample": True,
    }

    # 预设 prompts
    prompts = [
        "你好，请介绍一下你自己。",
        "1+1等于几？",
    ]

    print("\n--- Batch Inference ---\n")
    for prompt in prompts:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                **gen_config,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(f"User: {prompt}")
        print(f"Assistant: {response}")
        print()

    # 流式对话模式
    print("=" * 40)
    print("Interactive Stream Chat (type 'exit' to quit)")
    print("=" * 40)

    history = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]

    while True:
        user_input = input("\nUser: ").strip()
        if user_input.lower() in ["exit", "quit", ""]:
            print("Bye!")
            break

        history.append({"role": "user", "content": user_input})
        text = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # 使用 streamer 实现逐字输出
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = dict(
            **model_inputs,
            **gen_config,
            streamer=streamer,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        print("Assistant: ", end="", flush=True)
        response_parts = []
        for new_text in streamer:
            print(new_text, end="", flush=True)
            response_parts.append(new_text)
        print()
        thread.join()

        response = "".join(response_parts)
        history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
