import datetime
import os

from bitsandbytes.optim import AdamW8bit
from datasets import load_dataset
from safetensors.torch import save_file
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from typing import Optional
from tqdm import tqdm
import wandb

from moondream.finetune.dataloader import get_dataloader
from moondream.finetune.lr_schedule import lr_schedule
from moondream.torch.image_crops import prepare_crops, reconstruct_from_crops

from ..torch.moondream import MoondreamConfig, MoondreamModel, text_encoder
from ..torch.text import TextConfig, _lm_head, _produce_hidden
from ..torch.weights import load_weights_into_model
from .strings import ANSWER_EOS, BOS_TOKEN, PAD_TOKEN
from .finetune_text_sweep import eval_model

USE_WANDB = False
VERBOSE = True
LR = 3e-6
EPOCHS = 1
GRAD_ACCUM_STEPS = 1
BATCH_SIZE = 2

HF_TOKEN = os.environ["HF_TOKEN"]
WANDB_API_KEY = os.environ["WANDB_API_KEY"]
MD_QUESTION = os.environ["MD_QUESTION"]
HF_DS_REPO = os.environ["HF_DS_REPO"]
HF_DS_TARGET_COLUMN = os.environ["HF_DS_TARGET_COLUMN"]
BASEMODEL_PATH = os.environ["BASEMODEL_PATH"]
DEBUG = os.environ.get("DEBUG", "False").lower() in ["true", "1", "yes", "y"]


class CocoDataset(Dataset):
    def _tokenize_dataset(self, batch):
        self.tokenizer.no_padding()
        batch_size = next(len(v) for v in batch.values())
        questions = [f"\n\nQuestion: {MD_QUESTION}\n\nAnswer:"] * batch_size

        images = batch["image"]
        answers = batch[HF_DS_TARGET_COLUMN]

        def tokenize(input):
            encodings = self.tokenizer.encode_batch(input)
            return {
                "input_ids": [enc.ids for enc in encodings],
                "attention_mask": [enc.attention_mask for enc in encodings],
            }

        q_enc = tokenize(questions)
        a_enc = tokenize(answers)

        return {
            "image": images,
            "question_ids": q_enc["input_ids"],
            "answer_ids": a_enc["input_ids"],
            "q_attn_mask": q_enc["attention_mask"],
            "a_attn_mask": a_enc["attention_mask"],
        }

    def __init__(
        self, tokenizer: Optional[Tokenizer], filepath: Optional[str], split="train"
    ):
        if tokenizer is None and filepath is None:
            raise ValueError("Either tokenizer or filepath must be provided.")
        if tokenizer is None:
            raise NotImplementedError(
                "Tokenizer loading from file is not implemented yet."
            )
        else:
            ds = load_dataset(HF_DS_REPO, token=HF_TOKEN)[split]
            if DEBUG:
                ds = ds.select(range(128 * 2 + 1))
            self.tokenizer = tokenizer
            self.pad_token_id = tokenizer.token_to_id(PAD_TOKEN)
            self.data = ds.map(
                self._tokenize_dataset,
                batched=True,
                desc="Tokenizing dataset",
                remove_columns=ds.column_names,
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample


def text_loss(
    inputs_embeds: torch.Tensor,
    w: nn.Module,
    labels: torch.Tensor,  # labels are [B, max_answer_len], already shifted and padded with -100
    config: TextConfig,
    attention_mask: torch.Tensor = None,
):
    B, q_len_total, D_emb = inputs_embeds.shape
    hidden_BTC = _produce_hidden(
        inputs_embeds, w, config, attention_mask=attention_mask
    )
    lm_logits = _lm_head(hidden_BTC, w)  # Shape: [B, q_len_total, V]

    loss = None
    if labels is not None:
        # `labels` have shape [B, max_a_len].
        # We need to extract the logits from `lm_logits` that correspond to predicting these `labels`.
        # The answer embeddings are the last `labels.shape[1]` embeddings in `inputs_embeds`.
        # The logits for predicting the first token of `labels` (e.g., A2 if original answer was A1,A2,...)
        # are generated after seeing the input token corresponding to A1.
        # This input token (A1) is at index `q_len_total - labels.shape[1]` in `inputs_embeds`.
        # So the logits we need start at index `q_len_total - labels.shape[1]` in `lm_logits`.
        start_logit_idx_for_answer = q_len_total - labels.shape[1]

        # The slice should have the same length as `labels.shape[1]`.
        # So the end index for slicing (exclusive) is start_logit_idx_for_answer + labels.shape[1],
        # which simplifies to q_len_total.
        end_logit_idx_for_answer_slice = q_len_total

        logits_for_answers = lm_logits[
            :, start_logit_idx_for_answer:end_logit_idx_for_answer_slice, :
        ].contiguous()

        # Defensive check
        if logits_for_answers.shape[1] != labels.shape[1]:
            # This check should now pass.
            raise ValueError(
                f"Logits slice for answers shape mismatch with labels shape: "
                f"{logits_for_answers.shape[1]} vs {labels.shape[1]}. "
                f"lm_logits shape: {lm_logits.shape}, labels shape: {labels.shape}, "
                f"start_idx: {start_logit_idx_for_answer}, end_idx_slice: {end_logit_idx_for_answer_slice}"
            )

        loss = nn.CrossEntropyLoss(ignore_index=-100)(
            logits_for_answers.reshape(-1, logits_for_answers.size(-1)),
            labels.reshape(-1),
        )
    return loss


def main():
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your setup.")
        exit(1)
    # torch.set_default_device("cpu")

    if USE_WANDB:
        wandb.login(key=WANDB_API_KEY)
        wandb.init(
            project="moondream-ft",
            config={
                "EPOCHS": EPOCHS,
                "GRAD_ACCUM_STEPS": GRAD_ACCUM_STEPS,
                "LR": LR,
            },
        )

    config = MoondreamConfig()
    model = MoondreamModel(config)
    model.tokenizer.add_special_tokens([PAD_TOKEN])
    pad_token_id = model.tokenizer.token_to_id(PAD_TOKEN)
    bos_token_id = model.tokenizer.token_to_id(BOS_TOKEN)
    load_weights_into_model(BASEMODEL_PATH, model)

    optimizer = AdamW8bit(
        [
            {"params": model.text.parameters()},
        ],
        lr=LR,
        betas=(0.9, 0.95),
        eps=1e-6,
    )

    dataset = CocoDataset(tokenizer=model.tokenizer, split="train")
    dataloader = get_dataloader(
        dataset,
        pad_token_id=pad_token_id,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1,
    )

    total_steps = EPOCHS * len(dataloader) // GRAD_ACCUM_STEPS
    pbar = tqdm(total=total_steps)

    optimizer.zero_grad()
    global_step = 0

    model.to("cuda")
    model.train()

    for epoch in range(EPOCHS):
        for step, batch in enumerate(dataloader):
            images = batch["images"]
            q_ids = batch["question_ids"].to(model.device)
            a_ids = batch["answer_ids"].to(model.device)
            q_attn_mask = batch["q_attn_mask"].to(model.device)
            a_attn_mask = batch["a_attn_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            # → batched vision encoding
            # 1) collect crops for every image
            batch_crops = [
                prepare_crops(img, config.vision, device=model.device)
                for img in images
            ]
            # each element is (crops_tensor, tiling_info)
            # 2) cat all crops into one big [sum(N_i), …] tensor
            all_crops = torch.cat([c for c, _ in batch_crops], dim=0)

            # 3) single forward through vision encoder
            with torch.no_grad():
                vis_outs = model._vis_enc(all_crops)
            # unpack same as _run_vision_encoder does:
            global_feats = vis_outs[0]
            local_feats = vis_outs[1:].view(
                -1,
                config.vision.enc_n_layers,
                config.vision.enc_n_layers,
                config.vision.enc_dim,
            )

            # 4) split back per image
            counts = [crops.size(0) for crops, _ in batch_crops]
            g_splits = torch.split(global_feats, counts, dim=0)
            l_splits = torch.split(local_feats,  counts, dim=0)

            # 5) reconstruct & project each example
            img_embs = []
            for (g_i, l_i), (_, tiling) in zip(zip(g_splits, l_splits), batch_crops):
                recon = reconstruct_from_crops(
                    l_i,
                    tiling,
                    patch_size=1,
                    overlap_margin=config.vision.overlap_margin,
                )
                img_embs.append(model._vis_proj(g_i, recon))

            img_emb = torch.stack(img_embs, dim=0)  # [B, T_img, D]

            # Text embeddings with attention masks
            bos_token = torch.tensor([[bos_token_id]] * BATCH_SIZE, device=model.device)
            bos_emb = text_encoder(
                bos_token, model.text, attention_mask=None
            )  # No mask needed
            q_emb = text_encoder(
                q_ids, model.text, attention_mask=q_attn_mask
            )  # With mask
            a_emb = text_encoder(
                a_ids, model.text, attention_mask=a_attn_mask
            )  # With mask

            # Concatenate embeddings
            inputs_embeds = torch.cat([bos_emb, img_emb, q_emb, a_emb], dim=1)

            # Create combined attention mask for the full sequence
            # This ensures padding tokens in question/answer are ignored in attention
            combined_attn_mask = torch.cat(
                [
                    torch.ones(
                        BATCH_SIZE, img_emb.shape[1] + 1, device=model.device
                    ),  # BOS + image tokens
                    q_attn_mask,
                    a_attn_mask,
                ],
                dim=1,
            ).to(dtype=torch.bool)

            # Compute loss with attention_mask
            loss = text_loss(
                inputs_embeds=inputs_embeds,
                w=model.text,
                labels=labels,
                config=config.text,
                attention_mask=combined_attn_mask,
            )

            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()

            # Optimizer step
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # Update learning rate
                lr = lr_schedule(LR, global_step, total_steps)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                pbar.set_postfix(
                    {"step": global_step, "loss": loss.item() * GRAD_ACCUM_STEPS}
                )
                pbar.update(1)
                if USE_WANDB:
                    wandb.log({"loss/train": loss.item() * GRAD_ACCUM_STEPS, "lr": lr})

    if USE_WANDB:
        wandb.finish()

    print(eval_model(model))
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # save_file(
    #     model.state_dict(),
    #     f"/moondream/models/moondream_finetune_{timestamp}.safetensors",
    # )


if __name__ == "__main__":
    """
    To run: python -m moondream.finetune.finetune_text_multibatch
    """
    main()
