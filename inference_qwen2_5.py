import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    # 模型路径
    model_path = os.path.expanduser("E:/models/Qwen2.5-0.5B")

    # 加载 tokenizer
    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 加载模型
    print(f"Loading model from: {model_path}")
    llm = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
    )

    # 采样参数
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
    )

    # 输入提示
    prompts = [
        "你好，请介绍一下你自己。",
        "1+1等于几？",
    ]

    # 应用 chat_template 格式化对话
    # Qwen2.5 支持多轮对话，这里演示单轮
    formatted_prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    print("\nGenerating outputs...\n")
    outputs = llm.generate(formatted_prompts, sampling_params)

    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        print(f"--- Prompt {i+1} ---")
        print(f"User: {prompt}")
        print(f"Assistant: {output['text']}")
        print()

    # 交互式推理模式
    print("=" * 40)
    print("Interactive mode (type 'exit' to quit)")
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

        prompt_text = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
        )

        result = llm.generate([prompt_text], sampling_params)[0]
        assistant_response = result["text"]

        print(f"Assistant: {assistant_response}")
        history.append({"role": "assistant", "content": assistant_response})


if __name__ == "__main__":
    main()
