import argparse
import os
import torch

from task1_finetuning.utils import set_seed
from task1_finetuning.data import prepare_and_cache_dataset
from task1_finetuning.model import load_qwen_model, configure_lora, convert_model_to_oplora
from task1_finetuning.trainer import PA3Trainer
from task1_finetuning.logger import TrainingLogger

def parse_args():
    parser = argparse.ArgumentParser(description="PA3 Task 1 Model Fine-Tuning Entry Point")
    parser.add_argument("--variant", type=str, required=True, choices=["full_ft", "l2_sp", "lora_r4", "lora_r16", "oplora_r16"])
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="data/metamath_tokenized")
    parser.add_argument("--output_root", type=str, default="./checkpoints/task1")
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lambda_l2sp", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--max_updates", type=int, default=1250)
    
    args = parser.parse_args()
    
    if args.variant in ["full_ft", "l2_sp"]:
        if args.learning_rate is None:
            args.learning_rate = 2e-5
    else:
        if args.learning_rate is None:
            args.learning_rate = 2e-4
            
    if args.variant == "l2_sp" and args.lambda_l2sp is None:
        args.lambda_l2sp = 1e-3
    elif args.variant != "l2_sp":
        args.lambda_l2sp = 0.0

    return args

def main():
    set_seed(6304)
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    output_dir = os.path.join(args.output_root, args.variant)
    os.makedirs(output_dir, exist_ok=True)

    model, tokenizer = load_qwen_model(args.model_name, dtype=torch.float16)
    tokenized_dataset = prepare_and_cache_dataset(tokenizer=tokenizer, cache_dir=args.data_dir)

    initial_params = None
    if args.variant == "full_ft":
        for p in model.parameters():
            p.requires_grad = True
            p.data = p.data.to(torch.float32)  # FP32 master parameters for Full FT
    elif args.variant == "l2_sp":
        for p in model.parameters():
            p.requires_grad = True
            p.data = p.data.to(torch.float32)  # FP32 master parameters for L2-SP
        initial_params = {n: p.detach().clone().to(torch.float16).cpu() for n, p in model.named_parameters()}
    elif args.variant == "lora_r4":
        model = configure_lora(model, r=4, alpha=8)
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.to(torch.float32)  # FP32 adapters, frozen FP16 backbone
    elif args.variant == "lora_r16":
        model = configure_lora(model, r=16, alpha=32)
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.to(torch.float32)
    elif args.variant == "oplora_r16":
        model = convert_model_to_oplora(model, r=16, alpha=32, k=128)
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.to(torch.float32)

    logger = TrainingLogger(output_dir=output_dir, variant=args.variant)
    trainer = PA3Trainer(
        model=model, tokenizer=tokenizer, train_dataset=tokenized_dataset,
        output_dir=output_dir, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, lambda_l2sp=args.lambda_l2sp,
        initial_params=initial_params, device=device,
        logger=logger,
    )
    torch.cuda.empty_cache()
    trainer.train(num_epochs=1, batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum_steps, max_updates=args.max_updates, resume_from_checkpoint_path=args.checkpoint)

if __name__ == "__main__":
    main()