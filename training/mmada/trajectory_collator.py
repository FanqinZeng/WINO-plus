import torch
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class TrajectoryDataCollator:
    pad_token_id: int
    ignore_id: int = -100
    max_length: int = 2048

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        target_length = min(self.max_length, max([len(f["input_ids"]) for f in features]))

        padded_input_ids = []
        padded_labels = []
        padded_target_masks = []

        for feature in features:
            inp = feature["input_ids"]
            lbl = feature["labels"]
            msk = feature["target_mask"]

            # Truncation
            inp = inp[:target_length]
            lbl = lbl[:target_length]
            msk = msk[:target_length]

            # Padding
            pad_len = target_length - len(inp)
            if pad_len > 0:
                p_inp = inp + [self.pad_token_id] * pad_len
                p_lbl = lbl + [self.ignore_id] * pad_len
                p_msk = msk + [False] * pad_len
            else:
                p_inp = inp
                p_lbl = lbl
                p_msk = msk

            padded_input_ids.append(torch.tensor(p_inp, dtype=torch.long))
            padded_labels.append(torch.tensor(p_lbl, dtype=torch.long))
            padded_target_masks.append(torch.tensor(p_msk, dtype=torch.bool))

        prompt_lengths = torch.tensor([f["prompt_lengths"] for f in features], dtype=torch.long)
        block_nums = torch.tensor([f["block_num"] for f in features], dtype=torch.long)
        unique_ids = [f.get("unique_id", "") for f in features]

        batch = {
            "input_ids": torch.stack(padded_input_ids),
            "labels": torch.stack(padded_labels),
            "target_mask": torch.stack(padded_target_masks),
            "prompt_lengths": prompt_lengths,
            "block_num": block_nums,
            "unique_ids": unique_ids
        }

        return batch