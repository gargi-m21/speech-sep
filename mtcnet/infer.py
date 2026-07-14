import argparse
import os

import torch
import torchaudio

from mtcnet.config import MTCNetConfig
from mtcnet.model import MTCNet


def load_audio(path: str, sample_rate: int):

    waveform, sr = torchaudio.load(path)

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != sample_rate:

        resampler = torchaudio.transforms.Resample(
            sr,
            sample_rate,
        )

        waveform = resampler(waveform)

    return waveform.squeeze(0)


def main():

    parser = argparse.ArgumentParser(
        description="MTC-Net Inference"
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--input",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--output_dir",
        default="outputs",
        type=str,
    )

    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
    )

    args = parser.parse_args()

    device = torch.device(
        args.device
        if torch.cuda.is_available()
        else "cpu"
    )

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    config = MTCNetConfig()

    model = MTCNet(config)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(
        checkpoint,
        strict=True,
    )

    model.to(device)

    model.eval()

    waveform = load_audio(
        args.input,
        config.training.sample_rate,
    )

    mixture = waveform.unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(mixture)

    separated = outputs["waveforms"][0]

    predicted_count = int(
        outputs["predicted_count"][0].item()
    )

    probs = outputs["presence_probs"][0]

    print("=" * 50)
    print("MTC-Net Inference")
    print("=" * 50)
    print()

    print(
        f"Predicted Speakers : {predicted_count}"
    )

    print(
        "Presence Probabilities:"
    )

    for i, p in enumerate(probs):

        print(
            f"Speaker {i+1}: {p.item():.4f}"
        )

    print()

    for i in range(predicted_count):

        audio = separated[i].detach().cpu()

        filename = os.path.join(
            args.output_dir,
            f"speaker_{i+1}.wav",
        )

        torchaudio.save(
            filename,
            audio.unsqueeze(0),
            config.training.sample_rate,
        )

        print(
            f"Saved {filename}"
        )

    print()
    print("Inference completed successfully.")


if __name__ == "__main__":
    main()