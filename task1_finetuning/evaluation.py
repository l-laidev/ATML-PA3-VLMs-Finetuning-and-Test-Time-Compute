from pathlib import Path


def evaluate_model_on_gsm8k(model_path: str, base_path: str | Path):
    base_path = Path(base_path) if not isinstance(base_path, Path) else base_path
    print(f"Running evaluation for checkpoint: {model_path}")
    
    # command structure for lm-eval:
    # lm_eval --model hf --model_args pretrained={model_path},dtype=float16 --tasks gsm8k --device cuda:0 --batch_size 4
    
    import subprocess
    cmds = [
        # GSM8K
        [
                "lm_eval", "run",
                "--model", "hf",
                "--model_args", f"pretrained={model_path},dtype=float16,trust_remote_code=True,enable_thinking=False",
                "--gen_kwargs", "temperature=0.0,top_p=1.0,max_new_tokens=1024",
                "--apply_chat_template",
                "--tasks", "gsm8k_cot",
                "--num_fewshot", "0",  # zero-shot
                "--seed", "6304",
                "--device", "cuda:0",
                "--batch_size", "16",
                "--output_path", str(base_path / "gsm8k_cot.json"),
                "--log_samples",
        ],
        # others
        [
                "lm_eval", "run",
                "--model", "hf",
                "--model_args", f"pretrained={model_path},dtype=float16,trust_remote_code=True,enable_thinking=False",
                "--tasks", "hellaswag,boolq,arc_easy",
                "--num_fewshot", "0",  # zero-shot
                "--seed", "6304",
                "--device", "cuda:0",
                "--batch_size", "16",
                "--output_path", str(base_path / "hellaswag_boolq_arc-easy.json"),
                "--log_samples",
        ]
    ]

    for cmd in cmds:
        print(f"Executing: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print("Evaluation completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error during lm-eval execution: {e}")
            print("Ensure lm-evaluation-harness is installed (`uv pip install lm-eval`).")
