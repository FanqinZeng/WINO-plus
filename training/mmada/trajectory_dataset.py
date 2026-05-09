from torch.utils.data import Dataset
from datasets import load_dataset

from .trajectory_utils import process_trajectory_batch

class TrajectoryDataset(Dataset):
    def __init__(self, data_path, mask_token_id=126336, block_length=128, trajectory_mode="original"):
        """
        data_path: path to the trajectory jsonl file
        trajectory_mode: "original" uses raw trajectory; "random" randomizes within blocks (ablation)
        """
        print(f"Loading trajectory dataset from {data_path}...")
        print(f"Trajectory mode: {trajectory_mode}")
        raw_dataset = load_dataset("json", data_files=data_path, split="train")

        print("Expanding trajectory dataset...")
        self.dataset = raw_dataset.map(
            process_trajectory_batch,
            batched=True,
            batch_size=1000,
            with_indices=True,
            remove_columns=raw_dataset.column_names,
            fn_kwargs={"mask_token_id": mask_token_id, "block_length": block_length, "trajectory_mode": trajectory_mode}
        )
        print(f"Expanded dataset size: {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        input_ids = item["input_ids"]
        labels = item["labels"]
        target_mask = item["target_mask"]
        prompt_lengths = item["prompt_lengths"]
        block_num = item["block_num"]
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "target_mask": target_mask,
            "prompt_lengths": prompt_lengths,
            "block_num": block_num,
            "unique_id": item.get("unique_id", "")
        }
