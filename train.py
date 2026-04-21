import torch

from nano_llama import NanoLlama


def main():
    model = NanoLlama()

    x = torch.randint(0, 100, (1, 10))
    out = model(x)

    print("Output shape:", out.shape)


if __name__ == "__main__":
    main()
