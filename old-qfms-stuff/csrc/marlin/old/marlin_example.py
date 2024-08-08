import torch, fms.utils.csrc.marlin as marlin
in_f = 1024
out_f = 2048
l_B = torch.empty((in_f // 16, out_f * 16 // 8), dtype=torch.int, device='cuda')
l_s = torch.empty((1, out_f), dtype=torch.half, device='cuda')
l_workspace = torch.zeros(out_f // 128 * 16, dtype=torch.int, device='cuda')
l_B, l_s = marlin.pack(torch.ones((out_f, in_f), dtype=torch.float16, device='cuda'), torch.ones(out_f, dtype=torch.float16, device='cuda'))

x = torch.eye(in_f, dtype=torch.float16, device='cuda')
C = torch.empty(x.shape[:-1] + (l_s.shape[1],), dtype=x.dtype, device=x.device)
marlin.mul(x, l_B, C, l_s, l_workspace)
C