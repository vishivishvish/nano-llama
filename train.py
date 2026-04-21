import torch

from nano_llama import Config, NanoLlama


def main():
    config = Config()
    model = NanoLlama(config)

    x = torch.randn(1, 10, config.dim)
    out = model(x)

    print("Output shape:", out.shape)


if __name__ == "__main__":
    main()
