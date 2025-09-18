import torch

def encoder(x, base, target_length, seq_length):
    with torch.no_grad():
        if target_length == 1:
            return x
        
        x_shape = x.shape
        if len(x_shape) > 2:
            B, c, H, W = x_shape
            x = x.permute(0, 2, 3, 1).flatten(start_dim=1)
        else:
            x = x.reshape(x.shape[0], -1)

        mask = base ** torch.arange(target_length - 1, -1, -1, device=x.device, dtype=x.dtype)
        digits = (x.unsqueeze(-1) // mask) % base
        x_code = digits.long()

        if len(x_shape) > 2:
            x_code = x_code.reshape(B, H, W, c * target_length).permute(0, 3, 1, 2)
        else:
            x_code = x_code.reshape(x.shape[0], target_length * x.shape[1])

        return x_code
    
def decoder(b, base, target_length, seq_length):
    with torch.no_grad():
        if target_length == 1:
            return b
        
        b_shape = b.shape

        if len(b_shape) > 2:
            B, c, H, W = b_shape
            b = b.permute(0, 2, 3, 1).flatten(start_dim=1)
    
        b = b.reshape(b.shape[0], seq_length, target_length)
        mask = base ** torch.arange(target_length - 1, -1, -1, 
                                    device=b.device, dtype=b.dtype)
        output = (b * mask).sum(dim=-1)

        if len(b_shape) > 2:
            output = output.reshape(B, H, W, c // target_length).permute(0, 3, 1, 2)

        return output

def super_encode(x, seq_length=3072):
    x_16Mbase = decoder(x, base=256, target_length=3, seq_length=seq_length//3)
    x_4096base = encoder(x_16Mbase, base=4096, target_length=2, seq_length=seq_length//3*2)
    return x_4096base

def super_decode(x_4096base, seq_length=3072):
    x_16Mbase = decoder(x_4096base, base=4096, target_length=2, seq_length=seq_length//3)
    x_256Mbase = encoder(x_16Mbase, base=256, target_length=3, seq_length=seq_length//3*3)
    return x_256Mbase

def convert_to_marginal_time(p, masks):
    p_shape = p.shape

    if masks is None:
        if len(p_shape) > 3:
            B, c, H, W, _ = p.shape
            padding = torch.zeros((B, c, H, W, 1), device=p.device)
            return torch.cat([p, padding], dim=4)
        else:
            B, L, C = p.shape
            padding = torch.zeros((B, L, 1), device=p.device)
            return torch.cat([p, padding], dim=2)

    
    l_L, _, C = masks.shape
    if len(p_shape) > 3:
        B, c, H, W, _ = p_shape
        p = p.permute(0, 2, 3, 1, 4).reshape(B, H * W * c, C)

    B, L, C = p.shape
    l = l_L // L

    expanded_p = p.unsqueeze(1).repeat(1, l, 1, 1).reshape(B, l_L, C)
    new_prob = (expanded_p.unsqueeze(2) * masks.unsqueeze(0).float()).sum(dim=3)
    extra_prob = torch.zeros((B, l_L, 1)).to(new_prob.device)
    new_prob = torch.cat([new_prob, extra_prob], dim=2)
    b = new_prob.shape[2]

    index_permute = torch.arange(L).repeat_interleave(l) + torch.arange(l).repeat(L) * L
    new_prob = new_prob[:, index_permute, :]

    if len(p_shape) > 3:
        new_prob = new_prob.reshape(B, H, W, c*l, b).permute(0, 3, 1, 2, 4)

    return new_prob

def convert_to_marginal_space(p, masks):
    p_shape = p.shape

    if masks is None:
        if len(p_shape) > 3:
            B, c, H, W, _ = p.shape
            padding = torch.zeros((B, c, H, W, 1), device=p.device)
            return torch.cat([p, padding], dim=4)
        else:
            B, L, C = p.shape
            padding = torch.zeros((B, L, 1), device=p.device)
            return torch.cat([p, padding], dim=2)

    l_L, b, C = masks.shape
    if len(p_shape) > 3:
        B, c, H, W, _ = p_shape
        p = p.permute(0, 2, 3, 1, 4).reshape(B, H * W * c, C)

    B, L, C = p.shape
    l = l_L // L

    expanded_p = p.unsqueeze(1).repeat(1, l, 1, 1).reshape(B, l * L, C)
    new_prob = torch.zeros((B, l * L, b+1), dtype=p.dtype, device=p.device)

    for j in range(b):
        new_prob[:, :, j] = (expanded_p * masks[:, j, :].float()).sum(dim=2)

    index_permute = torch.arange(L).repeat_interleave(l) + torch.arange(l).repeat(L) * L
    new_prob = new_prob[:, index_permute, :]

    if len(p_shape) > 3:
        new_prob = new_prob.reshape(B, H, W, c*l, b+1).permute(0, 3, 1, 2, 4)

    return new_prob

def create_p_mask(encoder, L, C, l, b, device):
    if l == 1:
        return None
    class_indices = torch.arange(C).view(C, 1).to(device)
    subtokenized = encoder(class_indices, 1).squeeze()
    subtokenized = subtokenized.unsqueeze(0).repeat(L, 1, 1)
    subtokenized = subtokenized.permute(2, 0, 1).reshape(l * L, C)
    masks = torch.zeros((l * L, b, C), dtype=torch.bool, device=device)
    masks.scatter_(1, subtokenized.unsqueeze(1), True)
    return masks

