import datetime
import math
import os

from bitsandbytes.optim import AdamW8bit
from datasets import load_dataset
from safetensors.torch import save_file
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
import wandb

from ..torch.moondream import MoondreamConfig, MoondreamModel, text_encoder
from ..torch.text import TextConfig, _lm_head, _produce_hidden
from ..torch.weights import load_weights_into_model
from .evaluate_finetune import rate_answer

# Your data should end with the eos token. Here is the textual representation.
ANSWER_EOS = "<|endoftext|>"

HF_TOKEN = os.environ["HF_TOKEN"]
WANDB_API_KEY = os.environ["WANDB_API_KEY"]
MD_QUESTION = os.environ["MD_QUESTION"]
HF_DS_REPO = os.environ["HF_DS_REPO"]
HF_DS_TARGET_COLUMN = os.environ["HF_DS_TARGET_COLUMN"]
BASEMODEL_PATH = os.environ["BASEMODEL_PATH"]
DEBUG = os.environ.get("DEBUG", "False").lower() in ["true", "1", "yes", "y"]
VERBOSE = os.environ.get("VERBOSE", "False").lower() in [
    "true", "1", "yes", "y"]


def lr_schedule(step, max_steps, base_lr, warmup_proportion, min_lr_factor):
    if max_steps == 0:
        return base_lr

    warmup_steps = warmup_proportion * max_steps

    if step < warmup_steps:
        if warmup_steps == 0:
            return base_lr  # Avoid division by zero if warmup_proportion is 0
        # Linear warmup from a small fraction (e.g., 0.1 * base_lr or min_lr_factor * base_lr) to base_lr
        warmup_start_lr = min_lr_factor * base_lr * 0.1  # Start very low
        current_warmup_step = step
        lr = warmup_start_lr + (base_lr - warmup_start_lr) * \
            (current_warmup_step / warmup_steps)
        return lr
    else:
        # Cosine decay from base_lr to min_lr_factor * base_lr
        # Avoid division by zero if warmup_proportion is 1
        if (1 - warmup_proportion) * max_steps == 0:
            return min_lr_factor * base_lr
        progress = (step - warmup_steps) / \
            ((1 - warmup_proportion) * max_steps)
        progress = min(progress, 1.0)  # Clamp progress to [0, 1]

        decay_initial_lr = base_lr
        decay_final_lr = min_lr_factor * base_lr

        lr = decay_final_lr + 0.5 * \
            (decay_initial_lr - decay_final_lr) * \
            (1 + math.cos(math.pi * progress))
        return lr


def text_loss(
    inputs_embeds: torch.Tensor, w: nn.Module, labels: torch.Tensor, config: TextConfig
):
    _, q_len, _ = inputs_embeds.shape
    hidden_BTC = _produce_hidden(inputs_embeds, w, config)
    lm_logits = _lm_head(hidden_BTC, w)

    loss = None
    if labels is not None:
        _, _, l_len = labels.shape
        shift_index = (q_len - l_len) - 1
        shifted_logits = lm_logits[..., shift_index:-1, :].contiguous()
        shifted_labels = labels.contiguous()
        loss = nn.CrossEntropyLoss()(
            shifted_logits.view(-1, shifted_logits.size(-1)),
            shifted_labels.view(-1),
        )
    return loss


class DocciDataset(Dataset):
    def __init__(self, split="train"):
        self.data = load_dataset("google/docci", trust_remote_code=True)[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        description = sample["description"]
        return {
            "image": sample["image"],
            "qa": {
                "question": "\n\nQuestion: Describe this image.\n\nAnswer:",
                "answer": f"{description}{ANSWER_EOS}",
            },
        }


class CocoDataset(Dataset):
    def __init__(self, split='train'):
        self.data = load_dataset(
            HF_DS_REPO, token=HF_TOKEN)[split]
        if DEBUG:
            self.data = self.data.select(range(min(257, len(self.data))))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            "image": sample["image"],  # Should be a PIL image
            "qa":
                {
                    "question": f"\n\nQuestion: {MD_QUESTION}\n\nAnswer:",
                    "answer": f"{sample[HF_DS_TARGET_COLUMN]}{ANSWER_EOS}",
            }

        }


def eval_model(model):
    model.compile()
    dataset = load_dataset(HF_DS_REPO, token=HF_TOKEN, split="val")
    correct = 0
    total = 0
    results = []

    results_table = wandb.Table(
        columns=["id", "ground_truth", "model_answer", "score"])

    for row in tqdm(dataset, desc="Evaluation"):
        image = row["image"]
        question = MD_QUESTION
        answer = row[HF_DS_TARGET_COLUMN]
        model_answer = model.query(image, question)["answer"]
        score = rate_answer(answer, model_answer)

        results.append(
            {
                "question": question,
                "ground_truth": answer,
                "model_answer": model_answer,
                "score": score,
            }
        )

        results_table.add_data(
            row["id"],
            answer,
            model_answer,
            score,
        )

        total += 1
        if score > 0.8:
            correct += 1
        elif VERBOSE:
            print(f"Score: {score}")
            print(f"Image: {row['id']}")
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print(f"Model Answer: {model_answer}")
        if DEBUG:
            print(f"Correct: {correct}, Total: {total}")
            print(f"Accuracy: {correct * 100 / total:.2f}")
            print("---------\n\n")
        elif VERBOSE:
            print("---------\n\n")

        if DEBUG and total > 10:
            break

    return {
        "avg_score": sum([r["score"] for r in results]) / len(results),
        "min_score": min([r["score"] for r in results]),
        "max_score": max([r["score"] for r in results]),
        "total_count": total,
        "results": results,
        "results_table": results_table,
    }


def main():
    timing = {}
    print("Initializing...")
    ts = datetime.datetime.now()

    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    else:
        print("CUDA is not available. Please check your setup.")
        exit(1)

    config_defaults = {
        "LR": 3e-6,
        "EPOCHS": 1,
        "GRAD_ACCUM_STEPS": 128,
        "WARMUP_PROPORTION": 0.1,
        "MIN_LR_FACTOR": 0.1,
        "ADAM_BETA1": 0.9,
        "ADAM_BETA2": 0.95,
        "ADAM_EPS": 1e-6,
        "WEIGHT_DECAY": 0.01,
    }

    wandb.login(key=WANDB_API_KEY)
    run = wandb.init(
        config=config_defaults,  # wandb.config will have sweep params + defaults
    )
    cfg = wandb.config

    timing["init"] = datetime.datetime.now() - ts
    print(f"Done after {(datetime.datetime.now() - ts).total_seconds():.0f} seconds")
    print("Loading model, optimizer & ds...")
    ts = datetime.datetime.now()

    config = MoondreamConfig()
    model = MoondreamModel(config)
    load_weights_into_model(BASEMODEL_PATH, model)

    optimizer = AdamW8bit(
        [
            {"params": model.text.parameters()},
        ],
        lr=cfg.LR,
        betas=(cfg.ADAM_BETA1, cfg.ADAM_BETA2),
        eps=cfg.ADAM_EPS,
    )

    dataset = CocoDataset("train")

    print(f"Done after {(datetime.datetime.now() - ts).total_seconds():.0f} seconds")
    timing["load"] = datetime.datetime.now() - ts
    ts = datetime.datetime.now()
    print("Starting training...")

    processed_samples_count = len(dataset) * cfg.EPOCHS
    total_steps = processed_samples_count // cfg.GRAD_ACCUM_STEPS
    pbar = tqdm(total=total_steps)

    i = 0
    for epoch in range(cfg.EPOCHS):
        for sample in dataset:
            i += 1
            with torch.no_grad():
                img_emb = model._run_vision_encoder(sample["image"])
            bos_emb = text_encoder(
                torch.tensor([[model.config.tokenizer.bos_id]],
                             device=model.device),
                model.text,
            )
            question_tokens = model.tokenizer.encode(
                sample["qa"]["question"]).ids
            question_emb = text_encoder(
                torch.tensor([[question_tokens]], device=model.device),
                model.text,
            ).squeeze(0)
            answer_tokens = model.tokenizer.encode(sample["qa"]["answer"]).ids
            answer_emb = text_encoder(
                torch.tensor([[answer_tokens]], device=model.device),
                model.text,
            ).squeeze(0)
            inputs_embeds = torch.cat(
                [bos_emb, img_emb[None], question_emb, answer_emb], dim=1
            )
            loss = text_loss(
                inputs_embeds=inputs_embeds,
                w=model.text,
                labels=torch.tensor([[answer_tokens]], device=model.device),
                config=config.text,
            )

            loss.backward()

            if i % cfg.GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

                lr = lr_schedule(i / cfg.GRAD_ACCUM_STEPS, total_steps,
                                 cfg.LR, cfg.WARMUP_PROPORTION, cfg.MIN_LR_FACTOR)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                pbar.set_postfix(
                    {"step": i // cfg.GRAD_ACCUM_STEPS, "loss": loss.item()})
                pbar.update(1)
                wandb.log(
                    {"loss/train": loss.item(),
                     "lr": optimizer.param_groups[0]["lr"]}
                )

    pbar.close()
    print(f"Done after {(datetime.datetime.now() - ts).total_seconds():.0f} seconds")
    timing["train"] = datetime.datetime.now() - ts
    ts = datetime.datetime.now()


    # Evaluation
    print("Starting evaluation...")
    model.eval()
    with torch.no_grad():
        eval_results = eval_model(model)

    print(f"Done after {(datetime.datetime.now() - ts).total_seconds():.0f} seconds")
    timing["eval"] = datetime.datetime.now() - ts

    print(f"Evaluation Results: Avg Score: {eval_results['avg_score']:.4f}")

    wandb.log({
        "eval/avg_score": eval_results["avg_score"],
        "eval/min_score": eval_results["min_score"],
        "eval/max_score": eval_results["max_score"],
        "eval/total_samples": eval_results["total_count"],
        "eval/results_table": eval_results["results_table"],

        "timing/init": timing["init"].total_seconds(),
        "timing/load": timing["load"].total_seconds(),
        "timing/train": timing["train"].total_seconds(),
        "timing/eval": timing["eval"].total_seconds(),
        "timing/train_per_sample": timing["train"].total_seconds() / processed_samples_count,
        "timing/eval_per_sample": timing["eval"].total_seconds() / eval_results["total_count"],
        "timing/total": sum(timing.values()).total_seconds()
    })

    wandb.finish()

    # Save the model
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"moondream_finetune_{run.id}_{timestamp}.safetensors"
    model_save_dir = "/moondream/models"
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, model_filename)
    print(f"Saving model to {model_save_path}")
    save_file(
        model.state_dict(),
        model_save_path,
    )

if __name__ == "__main__":
    """
    To run: python -m moondream.finetune.finetune_text_sweep
    """
    main()
