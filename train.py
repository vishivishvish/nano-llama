import torch

from nano_llama import Config, NanoLlama


def main():
    config = Config()
    model = NanoLlama(config)

    x = torch.randint(0, config.vocab_size, (1, 10))
    out = model(x)

    print("Output shape:", out.shape)


if __name__ == "__main__":
    main()
