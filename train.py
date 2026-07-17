import torch
import torch.optim as optim

from mtcnet.config import MTCNetConfig
from mtcnet.model import MTCNet
from mtcnet.losses import MTCNetLoss

from src.dataset import get_dataloader
import os
import soundfile as sf
from itertools import permutations
def si_sdr(reference, estimation, eps=1e-8):
    """
    Scale-Invariant SDR.
    reference : [T]
    estimation: [T]
    """
    reference = reference - reference.mean()
    estimation = estimation - estimation.mean()

    alpha = torch.dot(estimation, reference) / (
        torch.dot(reference, reference) + eps
    )

    target = alpha * reference
    noise = estimation - target

    ratio = (target.pow(2).sum() + eps) / (noise.pow(2).sum() + eps)

    return 10 * torch.log10(ratio)


def pit_si_sdri(mixture, estimates, targets):
    """
    Computes best-permutation SI-SDR improvement.

    mixture : [T]
    estimates : [C_pred,T]
    targets : [C_gt,T]
    """

    C = min(estimates.size(0), targets.size(0))

    estimates = estimates[:C]
    targets = targets[:C]

    best = -1e9

    for perm in permutations(range(C)):

        score = 0

        for est_idx, tgt_idx in enumerate(perm):

            separated = si_sdr(
                targets[tgt_idx],
                estimates[est_idx]
            )

            mixture_score = si_sdr(
                targets[tgt_idx],
                mixture
            )

            score += separated - mixture_score

        score /= C

        if score > best:
            best = score

    return best.item()
def main():

    print("Initializing MTC-Net configurations...")
    config = MTCNetConfig()

    # Small verification model
    config.backbone.num_blocks = 2
    config.backbone.dim = 64
    config.encoder.out_channels = 64
    config.masking_head.out_dim = 64
    config.counting_head.ffn_dim = 256
    config.backbone.ffn_dim = 256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    print("Instantiating MTC-Net Model...")
    model = MTCNet(config).to(device)

    loss_fn = MTCNetLoss(
        lambda_count=config.loss.lambda_count,
        qmax=config.counting_head.qmax,
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.lr,
    )

    print("\n--- Model Architecture Summary ---")
    print(f"Encoder: Conv1d, out={config.encoder.out_channels}")
    print(f"Backbone: R={config.backbone.num_blocks}, shared dim={config.backbone.dim}")
    print(f"Counting Head: Qmax={config.counting_head.qmax}, layers={config.counting_head.num_layers}")
    print(f"Masking Head: slots={config.masking_head.qmax}")
    print("----------------------------------\n")

    ####################################################################
    # DATASET
    ####################################################################

    train_loader = get_dataloader(
        csv_path="/kaggle/input/datasets/bigoone/summer-proj-2/summer proj 2/data/MiniLibriMix/metadata/mixture_train_mix_both.csv",
        data_root="/kaggle/input/datasets/bigoone/summer-proj-2/summer proj 2/data",
        n_src=2,
        sample_rate=8000,
        segment=2.0,
        train=True,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_loss = float("inf")
    ####################################################################
    # TRAIN
    ####################################################################

    epochs = 50

    print("Starting training...\n")

    model.train()
    
    for epoch in range(epochs):

        running_loss = 0.0
        i=0
        print("length of trainloader", len(train_loader))
        for mixtures, targets in train_loader:

            mixtures = mixtures.to(device)          # [B,T]
            targets = targets.to(device)            # [B,C,T]

            # Loss expects list of tensors
            targets_list = [targets[i] for i in range(targets.size(0))]

            optimizer.zero_grad()

            outputs = model(mixtures)

            total_loss, sep_loss, count_loss = loss_fn(
                outputs,
                targets_list,
            )

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.training.clip_grad_norm,
            )

            optimizer.step()

            running_loss += total_loss.item()

            # predicted counts
            pred_counts = outputs["predicted_count"].detach().cpu().tolist()
            actual_counts = [targets.size(1)] * mixtures.size(0)

            print(
                f"Epoch {epoch+1}/{epochs} "
                f"Loss={total_loss.item():.4f} "
                f"Sep={sep_loss.item():.4f} "
                f"Count={count_loss.item():.4f}"
            )
            print(i+1,'done')
            i+=1
            print(
                f"GT Counts: {actual_counts} | Pred Counts: {pred_counts}"
            )
        avg_loss = running_loss / len(train_loader)

        print(
            f"\nEpoch {epoch+1} completed "
            f"Average Loss = {avg_loss:.4f}\n"
        )
        ############################################################
        # SAVE CHECKPOINT EVERY EPOCH
        ############################################################

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "config": config,
        }

        torch.save(
            checkpoint,
            os.path.join(
                checkpoint_dir,
                f"mtcnet_epoch_{epoch+1}.pth"
            )
        )

        ############################################################
        # SAVE BEST MODEL
        ############################################################

        if avg_loss < best_loss:

            best_loss = avg_loss

            torch.save(
                checkpoint,
                os.path.join(
                    checkpoint_dir,
                    "mtcnet_best.pth"
                )
            )

            print(
                f"✓ Best model saved "
                f"(Loss={best_loss:.4f})"
            )
    ############################################################
    # SAVE FINAL MODEL
    ############################################################

    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        os.path.join(
            checkpoint_dir,
            "mtcnet_final.pth"
        )
    )

    print("✓ Final model saved.")
    ############################################################
    # INFERENCE + SAVE AUDIO + SI-SDRi
    ############################################################

    print("\nRunning inference evaluation...\n")

    model.eval()

    save_dir = "outputs"

    os.makedirs(save_dir, exist_ok=True)

    metric_list = []

    with torch.no_grad():

        processed = 0

        for mixtures, targets in train_loader:

            mixtures = mixtures.to(device)
            targets = targets.to(device)

            outputs = model(mixtures)

            predicted = outputs["waveforms"]

            B = mixtures.size(0)

            for b in range(B):

                if processed == 5:
                    break

                mixture = mixtures[b].cpu()

                estimate = predicted[b].cpu()

                target = targets[b].cpu()

                ####################################################
                # Save separated waveforms
                ####################################################

                for k in range(estimate.size(0)):

                    sf.write(
                        os.path.join(
                            save_dir,
                            f"sample_{processed+1}_speaker_{k+1}.wav"
                        ),
                        estimate[k].numpy(),
                        8000,
                    )

                ####################################################
                # Evaluate SI-SDRi
                ####################################################

                sisdri = pit_si_sdri(
                    mixture,
                    estimate,
                    target,
                )

                metric_list.append(sisdri)

                print(
                    f"Sample {processed+1}: "
                    f"Predicted Speakers={estimate.size(0)} "
                    f" SI-SDRi={sisdri:.2f} dB"
                )

                processed += 1

            if processed == 5:
                break

    print("\n======================================")

    print(f"Average SI-SDRi : {sum(metric_list)/len(metric_list):.2f} dB")

    print("Separated files saved to:", save_dir)

    print("======================================")


if __name__ == "__main__":
    main()