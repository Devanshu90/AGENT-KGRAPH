import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def format_example(example):
    triples=[]

    for t in example["triples"]:
        triples.append({
            "subject":str(t["subject"]),
            "predicate":str(t["predicate"]),
            "object":str(t["object"]),
            "confidence":float(t.get("confidence",1.0))
        })

    target=json.dumps(
        triples,
        ensure_ascii=False
    )

    prompt=(
        "Extract factual knowledge graph triples from the passage.\n"
        "Return a JSON array only.\n"
        "Each object must contain subject, predicate, object and confidence.\n"
        "Use normalized predicates such as directed_by, written_by, "
        "born_in, located_in, starred_actors, release_year and has_genre.\n\n"
        f"Passage:\n{example['text']}\n\n"
        f"Output:\n{target}"
    )

    return prompt


def load_training_data(path):
    records=[]

    with open(
        path,
        "r",
        encoding="utf-8-sig"
    ) as f:
        for line in f:
            line=line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records


def build_dataset(path,tokenizer,max_length):
    records=load_training_data(path)

    texts=[
        format_example(record)
        for record in records
    ]

    dataset=load_dataset(
        "json",
        data_files=str(path),
        split="train"
    )

    def tokenize(example):
        text=format_example(example)

        result=tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False
        )

        result["labels"]=result["input_ids"].copy()

        return result

    return dataset.map(
        tokenize,
        remove_columns=dataset.column_names
    )


def train(
    data_path,
    output_dir,
    model_name,
    max_length,
    epochs,
    batch_size
):
    tokenizer=AutoTokenizer.from_pretrained(
        model_name
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token

    dtype=(
        torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )

    model=AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype
    )

    lora_config=LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj"
        ],
        bias="none"
    )

    model=get_peft_model(
        model,
        lora_config
    )

    model.print_trainable_parameters()

    dataset=build_dataset(
        data_path,
        tokenizer,
        max_length
    )

    training_args=TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        fp16=torch.cuda.is_available()
    )

    trainer=Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
    )

    trainer.train()

    model.save_pretrained(
        output_dir
    )

    tokenizer.save_pretrained(
        output_dir
    )

    print(
        f"LoRA adapter saved to: {output_dir}"
    )


def main():
    parser=argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        required=True
    )

    parser.add_argument(
        "--output",
        default="runs/adapters/extractor-lora"
    )

    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct"
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=512
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1
    )

    args=parser.parse_args()

    train(
        data_path=args.data,
        output_dir=args.output,
        model_name=args.model,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size
    )


if __name__=="__main__":
    main()
