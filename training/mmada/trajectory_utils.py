import numpy as np

def generate_random_trajectory(traj_arr, block_length=128):
    """
    Generate a random trajectory by shuffling step assignments within each block.

    For each block of `block_length` tokens, extracts the set of unique step values
    from the original trajectory, then randomly reassigns them to positions.
    All unique values are guaranteed to appear at least once.

    Args:
        traj_arr: numpy array of original trajectory step values
        block_length: block size for randomization granularity

    Returns:
        numpy array with same shape/dlength as traj_arr, with randomized step assignments
    """
    n = len(traj_arr)
    random_traj = np.empty(n, dtype=traj_arr.dtype)

    for block_start in range(0, n, block_length):
        block_end = min(block_start + block_length, n)
        block_len = block_end - block_start
        block = traj_arr[block_start:block_end]
        unique_values = np.unique(block)

        # Randomly fill the entire block by sampling from the unique value set
        random_block = np.random.choice(unique_values, size=block_len)

        # Ensure all unique values appear at least once
        num_unique = len(unique_values)
        if num_unique <= block_len:
            guarantee_positions = np.random.choice(block_len, size=num_unique, replace=False)
            random_block[guarantee_positions] = np.random.permutation(unique_values)

        random_traj[block_start:block_end] = random_block

    return random_traj

def process_trajectory_batch(examples, indices=None, mask_token_id=126336, block_length=128, trajectory_mode="original"):
    """
    Vectorized unfolding of Trajectory data.

    Args:
        trajectory_mode: "original" uses the raw trajectory from data;
                         "random" randomizes step assignments within each block (ablation).
    """
    new_batch = {
        "trajectory_id": [],
        "input_ids": [],
        "labels": [],
        "prompt_lengths": [],
        "target_mask": [],
        "block_num": [],
        "unique_id": [],
        "image": []
    }
    
    if indices is None:
        indices = list(range(len(examples["prompt_ids"])))

    for idx in range(len(examples["prompt_ids"])):
        if "correct" in examples and examples["correct"][idx] is False:
            continue
            
        prompt_ids = examples["prompt_ids"][idx]
        generated_ids = examples["generated_ids"][idx]
        if "trajectory_accepted" in examples:
            traj_key = "trajectory_accepted"
        elif "wino_trajectory" in examples:
            traj_key = "wino_trajectory"
        elif "trajectory_proposed" in examples:
            traj_key = "trajectory_proposed"
        trajectory = examples[traj_key][idx]
        
        unique_id = examples["unique_id"][idx] if "unique_id" in examples else ""
        image_val = examples["image"][idx] if "image" in examples else None
        
        current_traj_id = indices[idx]
        
        prompt_len = len(prompt_ids)
        min_len = min(len(generated_ids), len(trajectory))
        
        if min_len == 0: continue
        
        gen_arr = np.array(generated_ids[:min_len])
        traj_arr = np.array(trajectory[:min_len])

        if trajectory_mode == "random":
            traj_arr = generate_random_trajectory(traj_arr, block_length)
        
        max_step = traj_arr.max()
        steps = range(max_step + 1)
        
        full_target_ids = prompt_ids + gen_arr.tolist()
        
        prompt_mask = np.zeros(prompt_len, dtype=bool)
        
        for i in steps:
            keep_mask = traj_arr < i
            
            masked_gen_arr = np.full_like(gen_arr, mask_token_id)
            masked_gen_arr[keep_mask] = gen_arr[keep_mask]
            
            input_ids = prompt_ids + masked_gen_arr.tolist()
            
            current_step_mask_gen = (traj_arr == i)
            
            if not np.any(current_step_mask_gen):
                continue
                
            full_target_mask = np.concatenate([prompt_mask, current_step_mask_gen])
            
            relative_indices = np.where(current_step_mask_gen)[0]
            current_block_num = int(relative_indices[0] // block_length)
            
            new_batch["input_ids"].append(input_ids)
            new_batch["labels"].append(full_target_ids)
            new_batch["prompt_lengths"].append(prompt_len)
            new_batch["target_mask"].append(full_target_mask.tolist())
            new_batch["block_num"].append(current_block_num)
            new_batch["trajectory_id"].append(current_traj_id)
            new_batch["unique_id"].append(unique_id)
            new_batch["image"].append(image_val)
            
    return new_batch