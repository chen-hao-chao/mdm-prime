import torch

def encoder(x, base, target_length, seq_length):
    with torch.no_grad():
        if target_length == 1:
            return x
        
        x = x.reshape(x.shape[0], -1)
        mask = base ** torch.arange(target_length - 1, -1, -1, device=x.device, dtype=x.dtype)
        digits = (x.unsqueeze(-1) // mask) % base
        x_code = digits.long()
        x_code = x_code.reshape(x.shape[0], target_length * x.shape[1])
        return x_code
    

def decoder(b, base, target_length, seq_length):
    with torch.no_grad():
        if target_length == 1:
            return b
        b = b.reshape(b.shape[0], seq_length, target_length)
        mask = base ** torch.arange(target_length - 1, -1, -1, 
                                    device=b.device, dtype=b.dtype)
        output = (b * mask).sum(dim=-1)
        return output


def create_extended_p_filter(encoder, C, l, b):
    if l == 1:
        return None
    class_indices = torch.arange(C).view(C, 1)
    subtokenized = encoder(class_indices, 1).squeeze().transpose(0, 1)
    masks = (subtokenized.unsqueeze(2) == torch.arange(b).view(1, 1, b))
    return masks


def convert_to_marginal_filter_logit(logit, extended_masks, class_index):
    l, C, b = extended_masks.shape
    B, L, _ = logit.shape
    l_L = l * L

    assert class_index.shape[1] == l_L, "Class index shape must match the extended masks shape"

    extended_masks = extended_masks.to(logit.device)
    extra_masks = torch.ones((C, 1)).to(extended_masks.device)
    extended_class_index = class_index.view(B, l_L, 1, 1).expand(-1, -1, C, 1)

    with torch.no_grad():
        accumulated_mask = torch.zeros((B, L, C)).to(logit.device)
        for j in range(l):
            indices = torch.arange(L, device=logit.device) * l + j
            extended_class_index_ = extended_class_index[:, indices, :, :]
            extended_masks_ = extended_masks[j, :, :]
            new_masks = torch.cat([extended_masks_, extra_masks], dim=1)
            new_masks = new_masks.view(1, 1, C, b+1).expand(B, L, C, b+1)

            prob_masking = torch.gather(new_masks,
                        dim=3,
                        index=extended_class_index_)
            accumulated_mask += prob_masking.view(B, L, C)
    
    masked_logit = logit.masked_fill((accumulated_mask != l), float('-inf'))
    log_probs = torch.log_softmax(masked_logit, dim=-1)
    return log_probs
    