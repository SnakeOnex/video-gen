import torch, torch.nn as nn, torchvision, argparse, time, time, tqdm, PIL, wandb, torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader
from pathlib import Path
from dataset import VideoDataset, visualize_video

from vqvae import VQGAN, VQGANConfig
from lpips import LPIPS

device = "cuda"

def make_plot(x_orig, x_recon, nrow=8):
    x_comb = torch.cat([x_orig.cpu().detach(), x_recon.cpu().detach()], dim=2)
    x_comb = ((x_comb + 1) / 2)
    grid_image = torchvision.utils.make_grid(x_comb, nrow=nrow).permute(1, 2, 0)
    grid_image = (grid_image.numpy() * 255).astype("uint8")
    return PIL.Image.fromarray(grid_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dmlab")
    args = parser.parse_args()


    if args.dataset == "dmlab":
        dataset_path = Path("../teco/dmlab/train")
        config = VQGANConfig(
            num_codebook_vectors=512,
            latent_dim=8, 
            resolution=64, 
            ch_mult=(1, 2, 2, 2)
        )

    wandb.init(project="vqvae-video", 
               name=f"{args.dataset}-{int(time.time()):.0f}",
               config=config.__dict__)

    dataset = VideoDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    perceptual_loss_fn = LPIPS().eval().to(device)

    vqvae = VQGAN(config).to(device)
    optim = torch.optim.Adam(vqvae.parameters(), lr=1e-4)

    steps = 0
    best_loss = float("inf")
    for epoch in range(100):
        for i, (video, _) in enumerate(tqdm.tqdm(loader)):
            video = video.to(device)
            frame_idx = torch.randint(0, video.shape[1], (8,))
            frames = video[:,frame_idx,...]
            x = rearrange(frames, "b f c h w -> (b f) c h w")
            out, indices, quantize_loss = vqvae(x)

            l1_loss = abs(out - x)
            percep_loss = perceptual_loss_fn(out, x)
            recon_loss = (l1_loss + percep_loss).mean()
            train_loss = quantize_loss + recon_loss

            if steps > 1000 and train_loss.item() < best_loss:
                best_loss = train_loss.item()
                torch.save(vqvae.state_dict(), f"vqvae_{args.dataset}_best.pt")

            if steps % 10 == 0:
                wandb.log({"train/quantize_loss": quantize_loss.item(),
                           "train/recon_loss": recon_loss.item(),
                           "train/total_loss": train_loss.item()})
                if steps % 1000 == 0:
                    grid_image = make_plot(x, out)
                    wandb.log({"train/grid_image": wandb.Image(grid_image, caption="Original vs Reconstructed", file_type="jpg")})

            optim.zero_grad()
            train_loss.backward()
            optim.step()
            steps += 1
