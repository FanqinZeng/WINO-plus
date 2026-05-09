import torch
import torch.nn.functional as F

class TrajectoryTrainer:
    def __init__(self, mask_token_id=126336, block_size=128, w_ce_loss=1.0, w_unmask_loss=0.1, w_remask_loss=1.0, threshold=0.5, threshold_back=0.9):
        self.mask_token_id = mask_token_id
        self.block_size = block_size
        self.w_ce_loss = w_ce_loss
        self.w_unmask_loss = w_unmask_loss
        self.w_remask_loss = w_remask_loss
        self.threshold = threshold
        self.threshold_back = threshold_back

    def compute_loss(self, inputs, logits):
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        target_mask = inputs['target_mask']
        block_nums = inputs['block_num']
        prompt_lengths = inputs['prompt_lengths']

        device = input_ids.device

        # Global masks
        masked_indices = (input_ids == self.mask_token_id)
        rs_indices = masked_indices & (~target_mask)

        # Block-filtered masks for REMASK loss
        block_masked_indices = masked_indices.clone()
        block_rs_indices = rs_indices.clone()
        
        for i in range(len(block_nums)):
            prompt_len = prompt_lengths[i].item()
            block_num = block_nums[i].item()
            start = prompt_len + block_num * self.block_size
            end = prompt_len + (block_num + 1) * self.block_size
            
            block_masked_indices[i, :start] = False
            block_masked_indices[i, end:] = False
            
            block_rs_indices[i, :start] = False
            block_rs_indices[i, end:] = False

        num_mask = masked_indices.sum().item()
        num_remask = block_rs_indices.sum().item()

        # 1. CE Loss
        if self.w_ce_loss > 0 and target_mask.sum() > 0:
            masked_logits = logits[target_mask]
            masked_labels = labels[target_mask]
            token_loss = F.cross_entropy(masked_logits, masked_labels, reduction='none')
            ce_loss = token_loss.sum() / target_mask.sum()
        else:
            # Use torch.tensor(0.0, device=device, requires_grad=True) to avoid unnecessary computation graph
            ce_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # 2. Unmask Loss (Entropy)
        if self.w_unmask_loss > 0 and target_mask.sum() > 0:
            target_logits = logits[target_mask]
            with torch.no_grad():
                probs = F.softmax(target_logits, dim=-1)
                confidences, predictions = probs.max(dim=-1)
                target_labels = labels[target_mask]
                is_true = (predictions == target_labels)
                is_low_conf = (confidences < self.threshold_back)
                unmask_indices = is_true & is_low_conf

            if unmask_indices.sum() > 0:
                selected_logits = target_logits[unmask_indices]
                selected_probs = F.softmax(selected_logits, dim=-1)
                selected_log_probs = F.log_softmax(selected_logits, dim=-1)
                entropy = -torch.sum(selected_probs * selected_log_probs, dim=-1)
                unmask_loss = entropy.mean()
            else:
                unmask_loss = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            unmask_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # 3. Remask Loss (Negative Entropy) - Block-filtered
        if self.w_remask_loss > 0 and block_rs_indices.sum() > 0:
            rs_logits_all = logits[block_rs_indices]
            rs_labels_all = labels[block_rs_indices]

            with torch.no_grad():
                rs_probs_all = F.softmax(rs_logits_all, dim=-1)
                confidences, predictions = rs_probs_all.max(dim=-1)
                is_wrong = (predictions != rs_labels_all)
                is_high_conf = (confidences > self.threshold)
                target_error_indices = is_wrong & is_high_conf

            if target_error_indices.sum() > 0:
                selected_logits = rs_logits_all[target_error_indices]
                selected_probs = F.softmax(selected_logits, dim=-1)
                selected_log_probs = F.log_softmax(selected_logits, dim=-1)
                entropy = -torch.sum(selected_probs * selected_log_probs, dim=-1)
                remask_loss = -entropy.mean()
            else:
                remask_loss = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            remask_loss = torch.tensor(0.0, device=device, requires_grad=True)

        total_loss = (self.w_ce_loss * ce_loss +
                      self.w_unmask_loss * unmask_loss +
                      self.w_remask_loss * remask_loss)

        loss_dict = {
            'total_loss': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
            'ce_loss': ce_loss.item() if isinstance(ce_loss, torch.Tensor) else ce_loss,
            'unmask_loss': unmask_loss.item() if isinstance(unmask_loss, torch.Tensor) else unmask_loss,
            'remask_loss': remask_loss.item() if isinstance(remask_loss, torch.Tensor) else remask_loss,
            'num_masked': num_mask,
            'num_remasked': num_remask
        }

        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'unmask_loss': unmask_loss,
            'remask_loss': remask_loss,
            'loss_dict': loss_dict
        }