import torch
from torch.utils.data import DataLoader


def get_dataloader(dataset, pad_token_id, batch_size, shuffle=True, num_workers=1, **dataloader_args):
    def collate_fn(batch):
        images = [item["image"] for item in batch]
        question_ids = [item["question_ids"] for item in batch]
        answer_ids = [item["answer_ids"] for item in batch]
        q_attn_mask = [item["q_attn_mask"] for item in batch]
        a_attn_mask = [item["a_attn_mask"] for item in batch]
        # Pad sequences to the same length
        max_q_len = max(len(ids) for ids in question_ids)
        max_a_len = max(len(ids) for ids in answer_ids)
        question_ids = [ids + [pad_token_id] *
                        (max_q_len - len(ids)) for ids in question_ids]
        answer_ids = [ids + [pad_token_id] *
                      (max_a_len - len(ids)) for ids in answer_ids]
        q_attn_mask = [mask + [0] * (max_q_len - len(mask))
                       for mask in q_attn_mask]
        a_attn_mask = [mask + [0] * (max_a_len - len(mask))
                       for mask in a_attn_mask]
        # Convert to tensors
        question_ids = torch.tensor(question_ids)
        answer_ids = torch.tensor(answer_ids)
        q_attn_mask = torch.tensor(q_attn_mask)
        a_attn_mask = torch.tensor(a_attn_mask)
        # Shift labels for autoregressive prediction
        labels = answer_ids.clone()
        labels = labels.roll(shifts=-1, dims=1)  # Shift all tokens left
        labels[:, -1] = -100  # Last token has no next token to predict
        # Replace pads with ignore index
        labels[labels == pad_token_id] = (-100)
        return {
            "images": images,
            "question_ids": question_ids,
            "answer_ids": answer_ids,
            "labels": labels,
            "q_attn_mask": q_attn_mask,
            "a_attn_mask": a_attn_mask,
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        **dataloader_args
    )
