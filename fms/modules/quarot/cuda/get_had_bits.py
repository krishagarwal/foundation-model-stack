import scipy
import torch

def diag_tile_block(block, reps):
    assert block.shape[-1] == block.shape[-2]
    row = torch.nn.functional.pad(block, (0, block.shape[-1] * (reps - 1), 0, 0))
    return torch.concat(
        [torch.roll(row, block.shape[-1] * i, 1) for i in range(0, reps)]
    )

def get_strs(had_size):
    p1s = ["0b"] * 4
    p2s = ["0b"] * 4
    mask = ["0b"] * 4
    had = torch.tensor(scipy.linalg.hadamard(had_size))
    had16 = diag_tile_block(had, 16 // had_size)
    for t in range(32):
        row = t // 4
        col = t % 4
        for j in range(4):
            num1 = had16[row + (j % 2) * 8][(col + (j // 2) * 4) * 2]
            num2 = had16[row + (j % 2) * 8][(col + (j // 2) * 4) * 2 + 1]
            p1s[j] += "1" if num1 == 1 else "0" #if num1 == -1 else "n"
            p2s[j] += "1" if num2 == 1 else "0" #if num2 == -1 else "n"
            mask[j] += "1" if (num1 != 0 and num2 != 0) else "0" if (num1 == 0 and num2 == 0) else "x"
    print("{\n\t" + ",\n\t".join(p1s) + "\n}")
    print("{\n\t" + ",\n\t".join(p2s) + "\n}")
    print("{\n\t" + ",\n\t".join(mask) + "\n}")