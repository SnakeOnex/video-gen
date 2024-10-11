import torch, torch.nn as nn, torchvision, argparse, time, time, tqdm, PIL, wandb, torch.nn.functional as F
import numpy as np
from pathlib import Path
from einops import rearrange

from torch.utils.data import DataLoader
from dataset import VideoDataset, annotate_video, encode_video
from vqvae import VQGAN, VQGANConfig
# from st_transformer import STTransformer
# from st_transformer2 import STTransformer
from gpt import GPTLanguageModel, GPTConfig
from train_vqvae import make_plot

device = "cuda"

def make_video(x_orig, a_orig, x_gen, a_gen, output_path):
    # B x T x C x H x W inputs
    # T x C x N*H x 2*W output
    comps = []
    for b in range(x_orig.shape[0]):
        orig_vid = annotate_video(x_orig[b], a_orig[b])
        gen_vid = annotate_video(x_gen[b], a_gen[b].view(-1,1))
        comps.append(np.concatenate([orig_vid, gen_vid], axis=2))
    comps = np.concatenate(comps, axis=1)
    encode_video(comps, output_path)

def make_video_plot(x_orig, a_orig, x_gen, a_gen, nrow=8):
    grids = []
    for b in range(x_orig.shape[0]):
        orig_vid = annotate_video(x_orig[b], a_orig[b])
        gen_vid = annotate_video(x_gen[b], a_gen[b])
        orig_vid = rearrange(orig_vid, "t h w c -> h (t w) c")
        gen_vid = rearrange(gen_vid, "t h w c -> h (t w) c")
        x_comb = np.concatenate([orig_vid, gen_vid], axis=0)
        grids.append(x_comb)
    final_grid = np.concatenate(grids, axis=0)
    return PIL.Image.fromarray(final_grid)

@torch.no_grad()
def evaluate():
    test_loss = 0
    for i, (video, action) in enumerate(tqdm.tqdm(test_loader)):
        video = video.to(device)
        action = action.to(device)
        video = video[:,start:start+max_len,...]
        action = action[:,start:start+max_len].unsqueeze(-1) + codebook_size

        # tokenize the video
        video_vectorized = rearrange(video, "b f c h w -> (b f) c h w")
        _, indices, _ = vqvae.encode(video_vectorized)
        indices = rearrange(indices, "(b f h w) -> b f (h w)", b=bs, f=max_len, w=int(spatial_size**0.5), h=int(spatial_size**0.5))

        indices = torch.cat([indices, action], dim=2)

        tokens = rearrange(indices, "b f s -> b (f s)")
        x = tokens[:,:-1]
        targets = tokens[:,1:].contiguous().view(-1)
        logits, _ = st_transformer(x)
        logits = rearrange(logits, "b t c -> (b t) c")
        test_loss += F.cross_entropy(logits, targets).item()
    test_loss /= len(test_loader)
    wandb.log({"valid/loss": test_loss})

    return video, action, indices

@torch.no_grad()
def gen_action_conditioned(image_tokens, actions, out_path=None):
    image_tokens = image_tokens[:,0,:spatial_size]
    print(image_tokens.shape)

    tokens = image_tokens
    for action in actions:
        print('action: ', action)
        action = torch.zeros(tokens.shape[0],1, device=device, dtype=torch.int) + action
        tokens = torch.cat([tokens, action], dim=1)
        print(tokens.shape)
        tokens = st_transformer.generate(tokens, spatial_size)
        print(tokens.shape)
    action = torch.zeros(tokens.shape[0],1, device=device, dtype=torch.int) + action
    tokens = torch.cat([tokens, action], dim=1)

    gen_tokens = rearrange(tokens, "b (t s) -> (b t) s", t=max_len, s=(spatial_size+1))
    gen_video_tokens = torch.clip(gen_tokens[:,:spatial_size], 0, 511)
    gen_actions = torch.clip(gen_tokens[:,spatial_size]-codebook_size, 0, 2) 
    gen_actions = rearrange(gen_actions, "(b t) -> b t", b=vis_count, t=max_len)
    gen_video = vqvae.decode(gen_video_tokens)
    gen_video = rearrange(gen_video, "(b t) c h w -> b t c h w", b=vis_count, t=max_len)
    image = make_video_plot(video_orig, actions_orig, gen_video, gen_actions, nrow=max_len)
    wandb.log({f"act_cond={actions[0]}": wandb.Image(image,file_type="jpg")})
    if out_path is not None:
        image.save(out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dmlab")
    args = parser.parse_args()

    codebook_size = 512

    if args.dataset == "dmlab":
        dataset_path = Path("../teco/dmlab/")
        config = VQGANConfig(
            num_codebook_vectors=codebook_size,
            latent_dim=8, 
            resolution=64, 
            ch_mult=(1, 2, 2, 2)
        )

    start = 50
    max_len = 12
    bs = 8
    spatial_size = 64
    vis_count = 4
    cond_frames = [1, 2, 4]

    wandb.init(project="st-video", 
               name=f"{args.dataset}-{int(time.time()):.0f}",
               config=config.__dict__)

    train_set = VideoDataset(dataset_path / "train")
    test_set = VideoDataset(dataset_path / "test")

    print(f"loaded {len(train_set)} train and {len(test_set)} test videos")

    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=True, drop_last=True)

    vqvae = VQGAN(config).to(device)
    vqvae.load_state_dict(torch.load("vqvae_dmlab_best.pt", weights_only=True))
    # st_transformer = STTransformer(4, 256, 8, max_len=max_len, spatial_size=64).to(device)
    # gpt_config = GPTConfig(max_len*spatial_size+1, codebook_size+3, 1024, 16, 12, True, 0.1)
    gpt_config = GPTConfig(block_size=max_len*(spatial_size+1), 
                           vocab_size=codebook_size+3, 
                           n_embd=1024, 
                           n_head=16, 
                           n_layer=12, 
                           causal=True, 
                           dropout=0.1)
    st_transformer = GPTLanguageModel(gpt_config).to(device)
    optim = torch.optim.Adam(st_transformer.parameters(), lr=1e-4)

    steps = 0
    for epoch in range(100):
        for i, (video, action) in enumerate(tqdm.tqdm(train_loader)):
            video = video.to(device)
            action = action.to(device)
            video = video[:,start:start+max_len,...]
            action = action[:,start:start+max_len].unsqueeze(-1) + codebook_size

            # tokenize the video
            with torch.no_grad():
                video_vectorized = rearrange(video, "b f c h w -> (b f) c h w")
                _, indices, _ = vqvae.encode(video_vectorized)
                indices = rearrange(indices, "(b f h w) -> b f (h w)", b=bs, f=max_len, w=int(spatial_size**0.5), h=int(spatial_size**0.5))

            # append action tokens
            # b x f x 1 + b x f x s = b x f x (s+1)
            indices = torch.cat([indices, action], dim=2)
            

            tokens = rearrange(indices, "b f s -> b (f s)")
            x = tokens[:,:-1]
            targets = tokens[:,1:].contiguous().view(-1)

            logits, _ = st_transformer(x)
            # logits = rearrange(logits, "b f s c -> (b f s) c")
            # targets = rearrange(indices, "b f s -> (b f s)")
            logits = rearrange(logits, "b t c -> (b t) c")
            # targets = rearrange(indices, "b f s -> (b f s)")
            loss = F.cross_entropy(logits, targets)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if steps % 1000 == 0:
                video, action, indices = evaluate()
                with torch.no_grad():
                    for cond_frame in cond_frames:
                        frames_to_gen = max_len - cond_frame

                        # original video
                        video_orig = video[0:vis_count]
                        actions_orig = action[0:vis_count] - codebook_size

                        # reconstructed video
                        video_tokens = indices[0:vis_count]
                        tokens = video_tokens[:,:cond_frame,...]
                        tokens = tokens.view(vis_count, -1)
                        st_transformer.eval()
                        res = st_transformer.generate(tokens, (spatial_size+1)*frames_to_gen, verbose=True)
                        gen_tokens = rearrange(res, "b (t s) -> (b t) s", t=max_len, s=(spatial_size+1))
                        gen_video_tokens = torch.clip(gen_tokens[:,:spatial_size], 0, 511)
                        gen_actions = torch.clip(gen_tokens[:,spatial_size]-codebook_size, 0, 2) 
                        gen_actions = rearrange(gen_actions, "(b t) -> b t", b=vis_count, t=max_len)
                        gen_video = vqvae.decode(gen_video_tokens)
                        gen_video = rearrange(gen_video, "(b t) c h w -> b t c h w", b=vis_count, t=max_len)
                        image = make_video_plot(video_orig, actions_orig, gen_video, gen_actions, nrow=max_len)
                        make_video(video_orig, actions_orig, gen_video, gen_actions, f"gen_video_{cond_frame}.mp4")
                        # wandb log video
                        wandb.log({f"video cond={cond_frame}": wandb.Video(f"gen_video_{cond_frame}.mp4")})
                        wandb.log({f"cond={cond_frame}": wandb.Image(image,file_type="jpg")})

                    for action in range(3):
                        gen_action_conditioned(
                                video_tokens, 
                                [action for i in range(max_len-1)],
                                f"cond_video_{action}.jpg")


            if steps % 10 == 0:
                wandb.log({"train/loss": loss.item()})
            steps += 1

