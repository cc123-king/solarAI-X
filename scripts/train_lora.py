"""Validate separate SFT data by default; --run trains a local PEFT LoRA adapter."""

from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import random


def read_rows(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_data(train, validation, evaluation_manifest):
    train, validation = read_rows(train), read_rows(validation)
    if not train or not validation:
        raise ValueError("训练与验证数据都不能为空")
    manifest = json.loads(Path(evaluation_manifest).read_text(encoding='utf-8'))
    if not isinstance(manifest.get('fixture_groups'), list) or not manifest['fixture_groups']:
        raise ValueError('Supply an evaluation group manifest without prompts or reference answers')
    prohibited = set(manifest['fixture_groups'])
    groups = []
    for rows in [train, validation]:
        current = set()
        for row in rows:
            if row.get("usage") != "training_seed" or not row.get("fixture_group"):
                raise ValueError("仅接受单独构建的训练种子，禁止直接导入评测场景")
            if row["fixture_group"] in prohibited:
                raise ValueError("训练数据与评测夹具重叠")
            messages = row.get("messages", [])
            if (
                len(messages) < 3
                or messages[0].get("role") != "system"
                or messages[-1].get("role") != "assistant"
            ):
                raise ValueError("训练记录缺少系统指令或最终助手响应")
            if messages[-2].get("role") != "user":
                raise ValueError("本训练器只支持单次助手响应的SFT种子")
            if not (messages[-1].get("content") or messages[-1].get("tool_calls")):
                raise ValueError("训练目标不能为空")
            if any(
                not isinstance(m, dict)
                or m.get("role") not in {"system", "user", "assistant"}
                for m in messages
            ):
                raise ValueError("训练消息格式无效")
            current.add(row["fixture_group"])
        groups.append(current)
    if groups[0] & groups[1]:
        raise ValueError("训练与训练验证集的夹具组不能重叠")
    prompts = [
        json.dumps(row["messages"][:-1], ensure_ascii=False, sort_keys=True)
        for row in train + validation
    ]
    if len(prompts) != len(set(prompts)):
        raise ValueError("训练与验证记录存在重复输入")
    return train, validation


def encode_example(tokenizer, row, max_length):
    options = {"tokenize": True, "add_generation_prompt": False}
    if row.get("tools"):
        options["tools"] = row["tools"]
    full = tokenizer.apply_chat_template(row["messages"], **options)
    prompt = tokenizer.apply_chat_template(
        row["messages"][:-1], **dict(options, add_generation_prompt=True)
    )
    if full[: len(prompt)] != prompt:
        raise ValueError("Chat template前缀不一致，拒绝错误的loss mask")
    if len(full) > max_length:
        raise ValueError("样本超过max_length；请分段，不能静默截断训练目标")
    if len(full) <= len(prompt):
        raise ValueError("没有可训练的助手token")
    return {
        "input_ids": full,
        "attention_mask": [1] * len(full),
        "labels": [-100] * len(prompt) + full[len(prompt) :],
    }


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train", type=Path, default=root / "finetuning/train.seed.jsonl"
    )
    parser.add_argument(
        "--validation", type=Path, default=root / "finetuning/validation.seed.jsonl"
    )
    parser.add_argument('--evaluation-manifest', type=Path, required=True,
                        help='Group IDs only; this trainer never opens sealed evaluation cases')
    parser.add_argument(
        "--model",
        type=Path,
        help="Existing local HF model directory; downloads are disabled",
    )
    parser.add_argument("--base-model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    train, validation = validate_data(args.train, args.validation, args.evaluation_manifest)
    summary = {
        "train_seeds": len(train),
        "validation_seeds": len(validation),
        "declared_group_overlap": False,
        "status": "data_validated_training_not_started",
        "limitation": "Synthetic pipeline seeds; not a demonstrated quality-improving fine-tuning corpus.",
    }
    if not args.run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if (
        args.model is None
        or not (args.model / "config.json").is_file()
        or args.output is None
    ):
        parser.error("--run需要本地模型目录与新的输出目录")
    if args.output.exists():
        parser.error("输出目录已经存在，拒绝覆盖适配器或旧训练证据")
    if args.max_length < 64 or args.max_steps == 0 or args.max_steps < -1:
        parser.error("训练长度或步数无效")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model

    if not torch.cuda.is_available() and not args.allow_cpu:
        parser.error("当前PyTorch没有可用CUDA/ROCm设备；小模型CPU试跑需显式--allow-cpu")
    random.seed(42)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded_train = [encode_example(tokenizer, row, args.max_length) for row in train]
    encoded_validation = [
        encode_example(tokenizer, row, args.max_length) for row in validation
    ]
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False, torch_dtype=dtype
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        ),
    )
    model.enable_input_require_grads()
    model.peft_config["default"].base_model_name_or_path = args.base_model_id

    def collate(batch):
        size = max(len(row["input_ids"]) for row in batch)
        return {
            key: torch.tensor(
                [row[key] + [padding] * (size - len(row[key])) for row in batch],
                dtype=torch.long,
            )
            for key, padding in [
                ("input_ids", tokenizer.pad_token_id),
                ("attention_mask", 0),
                ("labels", -100),
            ]
        }

    arguments = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=1,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        bf16=bf16,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=1,
        report_to=[],
        seed=42,
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=encoded_train,
        eval_dataset=encoded_validation,
        data_collator=collate,
        processing_class=tokenizer,
    )
    outcome = trainer.train()
    validation_metrics = trainer.evaluate()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(args.output)
    manifest = dict(
        summary,
        status="training_completed",
        base_model_id=args.base_model_id,
        base_model_config_sha256=hashlib.sha256(
            (args.model / "config.json").read_bytes()
        ).hexdigest(),
        train_sha256=hashlib.sha256(args.train.read_bytes()).hexdigest(),
        validation_sha256=hashlib.sha256(args.validation.read_bytes()).hexdigest(),
        trainable_parameters=sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        total_parameters=sum(p.numel() for p in model.parameters()),
        train_metrics=outcome.metrics,
        validation_metrics=validation_metrics,
        versions={
            name: importlib.metadata.version(name)
            for name in ["torch", "transformers", "peft", "accelerate"]
        },
        heldout_agent_evaluation_completed=False,
    )
    (args.output / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
