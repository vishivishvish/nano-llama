import torch
import torch.nn as nn


class NanoLlama(nn.Module):
    def __init__(self):
        super().__init__()
        # placeholder
        pass

    def forward(self, x):
        return x


if __name__ == "__main__":
    model = NanoLlama()
    x = torch.randint(0, 100, (1, 10))
    out = model(x)
    print(out.shape)
