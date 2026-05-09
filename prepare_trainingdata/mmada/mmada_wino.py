import numpy as np
import torch
import torch.nn.functional as F


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


@torch.no_grad()
def mmu_generate_wino_with_trajectory(
    model,
    prompt: torch.Tensor,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    mask_id: int = 126336,
    threshold: float = 0.8,
    threshold_back: float = 0.9,
) -> tuple[torch.Tensor, int, list[int], list[int]]:
    """MMaDA WINO decoding with trajectory tracking for offline data collection.

    This mirrors the training-data patch used under
    code_revised/WINO-DLLM/prepare_trainingdata/ScienceQA/code/code_trajectory_record,
    but keeps the patched behavior local to prepare_trainingdata instead of changing
    the main MMaDA model code.
    """
    if gen_length % block_length != 0:
        raise ValueError("gen_length must be divisible by block_length")
    if prompt.shape[0] != 1:
        raise ValueError("Trajectory collection currently expects batch size 1.")

    device = prompt.device
    prompt_len = prompt.shape[1]
    x_block = torch.full((1, prompt_len + gen_length + block_length), mask_id, dtype=torch.long, device=device)
    x_block[:, :prompt_len] = prompt.clone()

    num_blocks = gen_length // block_length
    step = 0
    trajectory_accepted = [-1] * gen_length
    trajectory_proposed = [-1] * gen_length

    for num_block in range(num_blocks):
        block_step = 0
        current_start = prompt_len + num_block * block_length
        current_end = prompt_len + (num_block + 1) * block_length

        mask_index_block = x_block == mask_id
        mask_index_block[:, current_end:] = False

        unmask_index_block = torch.full_like(mask_index_block, False)
        unmask_index_block[:, -block_length:] = ~mask_index_block[:, current_start:current_end]

        position_ids = torch.cat(
            [
                torch.arange(prompt_len + gen_length, device=device),
                torch.arange(current_start, current_end, device=device),
            ]
        )
        attention_mask = torch.ones(1, 1, x_block.shape[1], x_block.shape[1], dtype=torch.bool, device=device)
        attention_mask[:, :, :, -block_length:] = False
        attention_mask[:, :, -block_length:, -block_length:] = True
        attention_mask[:, :, -block_length:, current_start:current_end] = ~torch.eye(
            block_length, dtype=torch.bool, device=device
        )
        last_accept = 30

        while mask_index_block.any():
            max_accept = min(max(int(mask_index_block.sum() * 0.7), 5), 20)
            logits = model(x_block, attention_mask=attention_mask, position_ids=position_ids).logits
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            unmask_index_shift_left = torch.zeros_like(unmask_index_block)
            unmask_index_shift_left[:, current_start:current_end] = unmask_index_block[:, -block_length:]
            x0[unmask_index_block] = x_block[unmask_index_shift_left]

            probs = F.softmax(logits.to(torch.float32), dim=-1)
            x0_p = torch.squeeze(torch.gather(probs, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            x0 = torch.where(mask_index_block, x0, x_block)
            confidence = torch.where(mask_index_block, x0_p, -np.inf)
            confidence_back = torch.where(unmask_index_block, x0_p, np.inf)

            transfer_index = confidence > threshold
            if transfer_index.sum() > max_accept:
                _, indices = torch.topk(confidence, k=max_accept, largest=True)
                transfer_index = torch.zeros_like(confidence, dtype=torch.bool)
                transfer_index.view(-1)[indices] = True
            elif not transfer_index.any():
                transfer_index.view(-1)[torch.argmax(confidence)] = True

            x_block[transfer_index] = x0[transfer_index]
            for pos_idx in range(transfer_index.shape[1]):
                if transfer_index[0, pos_idx]:
                    gen_pos = pos_idx - prompt_len
                    if 0 <= gen_pos < gen_length:
                        if trajectory_accepted[gen_pos] == -1:
                            trajectory_accepted[gen_pos] = step + block_step
                        if trajectory_proposed[gen_pos] == -1:
                            trajectory_proposed[gen_pos] = step + block_step

            num_accept = transfer_index.sum()

            if num_accept > 1:
                remask_index = confidence_back < threshold_back
                if remask_index.sum() >= last_accept:
                    num_remask = last_accept - 1
                    temp_mask = torch.zeros_like(confidence_back.view(-1), dtype=torch.bool)
                    _, indices = torch.topk(confidence_back.view(-1), k=num_remask, largest=False)
                    temp_mask[indices] = True
                    remask_index = temp_mask.view(confidence_back.shape)
            else:
                remask_index = torch.zeros_like(transfer_index)

            remask_index_shift = torch.zeros_like(remask_index)
            remask_index_shift[:, current_start:current_end] = remask_index[:, -block_length:]
            x_block[remask_index_shift] = mask_id
            for pos_idx in range(remask_index_shift.shape[1]):
                if remask_index_shift[0, pos_idx]:
                    gen_pos = pos_idx - prompt_len
                    if 0 <= gen_pos < gen_length:
                        trajectory_accepted[gen_pos] = -1

            mask_index_block[transfer_index] = False
            mask_index_block[remask_index_shift] = True
            block_step += 1

            transfer_index_shift = torch.zeros_like(transfer_index)
            transfer_index_shift[:, -block_length:] = transfer_index[:, current_start:current_end]
            unmask_index_block[transfer_index_shift] = True
            unmask_index_block[remask_index] = False
            last_accept = num_accept

        step += block_step

    return x_block[:, : prompt_len + gen_length], step, trajectory_accepted, trajectory_proposed

