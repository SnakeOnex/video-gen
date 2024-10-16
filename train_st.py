import torch, torch.nn as nn, torchvision, argparse, time, tqdm, PIL, wandb, torch.nn.functional as F
import numpy as np
from pathlib import Path
from einops import rearrange
from dataclasses import dataclass

from torch.utils.data import DataLoader
from dataset import VideoDataset, annotate_video, encode_video, action_to_text
from vqvae import VQGAN, VQGANConfig
from gpt import GPTLanguageModel, GPTConfig
from train_vqvae import make_plot


# video + action = b x f x (s + 1) tensor
# frame = b x s tensor
# action = b x 1 tensor
# next_image(image, action) = b x s tensor

device = "cuda" if torch.cuda.is_available() else "cpu"

def make_video(videos, output_path):
    # B x T x C x H x W inputs
    # T x C x N*H x 2*W output
    video_columns = []
    for i in range(len(videos)):
        video_rows = []
        video, actions = unpack_tokens(videos[i], vqvae)
        for b in range(videos[i].shape[0]):
            video_numpy = annotate_video(video[b], actions[b])
            video_rows.append(video_numpy)
        video_rows = np.concatenate(video_rows, axis=1)
        video_columns.append(video_rows)
    video_columns = np.concatenate(video_columns, axis=2)
    encode_video(video_columns, output_path)

def make_video_plot(videos, vqvae, nrow=8):
    video_image_row_sets = []
    for i in range(len(videos)):
        video, actions = unpack_tokens(videos[i], vqvae)
        video_image_rows = []
        print(video.shape, actions.shape)
        for b in range(video.shape[0]):
            video_numpy = annotate_video(video[b], actions[b])
            video_unrolled = rearrange(video_numpy, "t h w c -> h (t w) c")
            video_image_rows.append(video_unrolled)
        video_image_row_sets.append(video_image_rows)

    # connect the rows
    final_rows = []
    for i in range(len(video_image_row_sets[0])):
        for j in range(len(videos)):
            final_rows.append(video_image_row_sets[j][i])
    final_grid = np.concatenate(final_rows, axis=0)
    return PIL.Image.fromarray(final_grid)

@torch.no_grad()
def evaluate():
    test_loss = 0
    for i, (video, action) in enumerate(tqdm.tqdm(test_loader)):
        video = video.to(device)
        action = action.to(device)
        video = video[:,start:start+context_size,...]
        action = action[:,start:start+context_size].unsqueeze(-1) + codebook_size

        # tokenize the video
        video_vectorized = rearrange(video, "b f c h w -> (b f) c h w")
        _, indices, _ = vqvae.encode(video_vectorized)
        indices = rearrange(indices, "(b f h w) -> b f (h w)", b=video.shape[0], f=context_size, w=int(spatial_size**0.5), h=int(spatial_size**0.5))

        indices = torch.cat([indices, action], dim=2)

        tokens = rearrange(indices, "b f s -> b (f s)")
        x = tokens[:,:-1]
        targets = tokens[:,1:].contiguous().view(-1)
        logits, _ = st_transformer(x)
        logits = rearrange(logits, "b t c -> (b t) c")
        test_loss += F.cross_entropy(logits, targets).item()
    test_loss /= len(test_loader)
    wandb.log({"valid/loss": test_loss})
    return test_loss, video, action, indices

def pack_tokens(video, actions, vqvae, spatial_size=64, codebook_size=512):
    video_vectorized = rearrange(video, "b f c h w -> (b f) c h w")
    _, indices, _ = vqvae.encode(video_vectorized)
    video_tokens = rearrange(indices, "(b f h w) -> b f (h w)", b=video.shape[0], w=int(spatial_size**0.5), h=int(spatial_size**0.5))
    action_tokens = actions.unsqueeze(-1) + codebook_size
    return torch.cat([video_tokens, action_tokens], dim=2)

def unpack_tokens(tokens, vqvae, spatial_size=64, codebook_size=512):
    B, T, _ = tokens.shape
    video_tokens = torch.clip(tokens[:,:,:spatial_size], 0, codebook_size-1)
    video_tokens = rearrange(video_tokens, "b t s -> (b t) s")
    video = vqvae.decode(video_tokens)
    video = rearrange(video, "(b t) c h w -> b t c h w", b=B, t=T)
    actions = torch.clip(tokens[:,:,spatial_size]-codebook_size, 0, 2)
    return video, actions

@torch.no_grad()
def generate_video(cond_tokens, action_tokens, st_transformer, context_size, spatial_size=64):
    """
    cond_tokens: B x T_cond x (S+1) tensor
    action_tokens: B x T_gen tensor
    result: B x (T_cond+T_gen) x (S+1) tensor
    """
    T_cond, T_gen = cond_tokens.shape[1], action_tokens.shape[1]
    tokens = rearrange(cond_tokens, "b t s -> b (t s)")[:,:-1] # remove last action token
    for t in range(T_gen):
        action = action_tokens[:,t]
        tokens = torch.cat([tokens, action], dim=1)
        input_tokens = tokens[:,-(context_size-1)*(spatial_size+1):]
        new_tokens = st_transformer.generate(input_tokens, spatial_size)[:,-spatial_size:]
        print(f"generated {new_tokens.shape[1]} tokens, conditioned on {input_tokens.shape[1]} tokens")
        tokens = torch.cat([tokens, new_tokens], dim=1)
    tokens = torch.cat([tokens, action], dim=1)
    return rearrange(tokens, "b (t s) -> b t s", t=T_gen+T_cond, s=(spatial_size+1))

@dataclass
class TrainerConfig():
    gpt_config: GPTConfig
    gpt_path: str
    vqvae_config: VQGANConfig
    vqvae_path: str
    dataset_path: str
    context_size: int
    spatial_size: int
    gen_video_size: int
    batch_size: int
    lr: float
    vis_count: int
    cond_video_length: int
    cond_actions_length: int

    def __post_init__(self):
        self.spatial_dim = self.vqvae_config.spatial_dim

class Trainer():
    def __init__(self, config: TrainerConfig):
        for k, v in config.__dict__.items(): setattr(self, k, v)

        # 1. init models
        self.vqvae = VQGAN(self.config.vqvae_config).to(self.device).eval()
        self.vqvae.load_state_dict(torch.load(self.config.vqvae_path, weights_only=True))

        self.gpt = GPTLanguageModel(self.config.gpt_config).to(self.device)
        self.gpt.load_state_dict(torch.load(self.config.gpt_path, weights_only=True))

        # 2. init optimizers
        self.optim = self.configure_optimizers()

        # 3. init datasets
        self.train_loader = DataLoader(VideoDataset(self.config.dataset_path / "train"), batch_size=self.config.batch_size, shuffle=True, drop_last=True)
        self.test_loader = DataLoader(VideoDataset(self.config.dataset_path / "test"), batch_size=2*self.config.batch_size, shuffle=True, drop_last=True)

        # 4. running variables
        self.steps = 0
        self.best_loss = float("inf")

    def configure_optimizers(self):
        return torch.optim.Adam(self.gpt.parameters(), lr=self.config.lr)

    @torch.no_grad()
    def evaluate(self):
        test_loss = 0
        for i, (video, action) in enumerate(tqdm.tqdm(self.test_loader)):
            video, action = video.to(device), action.to(device)
            video = video[:,start:start+self.context_size,...]
            action = action[:,start:start+self.context_size].unsqueeze(-1) + self.codebook_size

            # tokenize the video
            video_vectorized = rearrange(video, "b f c h w -> (b f) c h w")
            _, indices, _ = self.vqvae.encode(video_vectorized)
            indices = rearrange(indices, "(b f h w) -> b f (h w)", b=video.shape[0], f=self.context_size, w=int(self.spatial_size**0.5), h=int(self.spatial_size**0.5))

            indices = torch.cat([indices, action], dim=2)

            tokens = rearrange(indices, "b f s -> b (f s)")
            x = tokens[:,:-1]
            targets = tokens[:,1:].contiguous().view(-1)
            logits, _ = self.gpt(x)
            logits = rearrange(logits, "b t c -> (b t) c")
            test_loss += F.cross_entropy(logits, targets).item()
        test_loss /= len(test_loader)
        wandb.log({"valid/loss": test_loss})
        return test_loss, video, action, indices

    def train(self):
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dmlab")
    parser.add_argument("--context_size", type=int, default=2)
    args = parser.parse_args()

    codebook_size = 512

    if args.dataset == "dmlab":
        dataset_path = Path("../teco/dmlab/")
        vqvae_config = VQGANConfig(
            num_codebook_vectors=codebook_size,
            latent_dim=8, 
            resolution=64, 
            ch_mult=(1, 2, 2, 2)
        )

        gpt_config = GPTConfig(block_size=args.context_size*(vqvae_config.spatial_dim+1), 
                               vocab_size=codebook_size+3, 
                               n_embd=368, 
                               n_head=4, 
                               n_layer=4, 
                               causal=True, 
                               dropout=0.1)

    config = TrainerConfig(gpt_config=gpt_config,
                           gpt_path="gpt_best.pt",
                           vqvae_config=vqvae_config,
                           vqvae_path="vqvae_best.pt",
                           dataset_path=dataset_path,
                           gen_video_size=12,
                           batch_size=8,
                           lr=1e-4,
                           vis_count=4,
                           cond_video_length=4,
                           cond_actions_length=10)

    trainer = Trainer(config)

    exit(0)

    start = 50
    context_size = 4 # in frames
    bs = 8
    spatial_size = 64
    # visualization params
    vis_count = 4
    gen_video_size = 12
    cond_video_length = 4 # how many frames to input as conditioning
    cond_actions_length = 10 # how many actions to input as conditioning

    wandb.init(project="st-video", 
               name=f"{args.dataset}-{int(time.time()):.0f}",
               config=vqgan_config.__dict__)

    train_set = VideoDataset(dataset_path / "train")
    test_set = VideoDataset(dataset_path / "test")

    print(f"loaded {len(train_set)} train and {len(test_set)} test videos")

    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=True, drop_last=True)

    vqvae = VQGAN(vqgan_config).to(device)
    vqvae.load_state_dict(torch.load("vqvae_dmlab_best.pt", weights_only=True))
    gpt_config = GPTConfig(block_size=context_size*(spatial_size+1), 
                           vocab_size=codebook_size+3, 
                           n_embd=368, 
                           n_head=4, 
                           n_layer=4, 
                           causal=True, 
                           dropout=0.1)
    st_transformer = GPTLanguageModel(gpt_config).to(device)
    optim = torch.optim.Adam(st_transformer.parameters(), lr=1e-4)

    steps = 0
    best_loss = float("inf")
    for epoch in range(100):
        for i, (video, action) in enumerate(tqdm.tqdm(train_loader)):
            video = video.to(device)
            action = action.to(device)
            start = torch.randint(0, video.shape[1]-context_size, (1,))
            video = video[:,start:start+context_size,...]
            action = action[:,start:start+context_size].unsqueeze(-1) + codebook_size

            # tokenize the video
            with torch.no_grad():
                video_vectorized = rearrange(video, "b f c h w -> (b f) c h w")
                _, indices, _ = vqvae.encode(video_vectorized)
                indices = rearrange(indices, "(b f h w) -> b f (h w)", b=bs, f=context_size, w=int(spatial_size**0.5), h=int(spatial_size**0.5))

            # append action tokens
            # b x f x 1 + b x f x s = b x f x (s+1)
            indices = torch.cat([indices, action], dim=2)
            tokens = rearrange(indices, "b f s -> b (f s)")
            x = tokens[:,:-1]
            targets = tokens[:,1:].contiguous().view(-1)

            logits, _ = st_transformer(x)
            logits = rearrange(logits, "b t c -> (b t) c")
            loss = F.cross_entropy(logits, targets)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if steps % 1000 == 0:
                test_loss, video, action, indices = evaluate()
                for i, (video, action) in enumerate(tqdm.tqdm(test_loader)):
                    video = video.to(device)
                    action = action.to(device)
                    start = torch.randint(0, video.shape[1]-cond_actions_length-cond_video_length, (1,))
                    video = video[:vis_count,start:start+cond_actions_length+cond_video_length]
                    action = action[:vis_count,start:start+cond_actions_length+cond_video_length]
                    orig_tokens = pack_tokens(video, action, vqvae)
                    break

                if test_loss < best_loss:
                    best_loss = test_loss
                    torch.save(st_transformer.state_dict(), "st_transformer_best.pt")
                    torch.save(vqvae.state_dict(), "vqvae_best.pt")
                else:
                    exit(0)

                with torch.no_grad():
                    action_tokens = orig_tokens[0:vis_count,-cond_actions_length:,spatial_size:spatial_size+1]
                    cond_tokens = orig_tokens[0:vis_count,:cond_video_length,:]

                    gen_tokens = generate_video(cond_tokens, action_tokens, st_transformer, context_size)
                    video_img = make_video_plot([orig_tokens, gen_tokens], vqvae, nrow=cond_video_length+cond_actions_length)
                    wandb.log({f"video cond={cond_video_length}": wandb.Image(video_img,file_type="jpg")})
                    make_video([orig_tokens, gen_tokens], "video.mp4")
                    wandb.log({f"video cond={cond_video_length}": wandb.Video("video.mp4")})

                    for action in range(3):
                        new_action_tokens = action_tokens * 0 + action + codebook_size
                        gen_tokens = generate_video(cond_tokens, new_action_tokens, st_transformer, context_size)
                        name = action_to_text(action)
                        video_img = make_video_plot([orig_tokens, gen_tokens], vqvae, nrow=cond_video_length+cond_actions_length)
                        wandb.log({f"action={name}": wandb.Image(video_img,file_type="jpg")})
                        make_video([orig_tokens, gen_tokens], f"video_{name}.mp4")
                        wandb.log({f"action={name}": wandb.Video(f"video_{name}.mp4")})

                    # alternating left and right
                    actions = [0, 0] + [1 if i % 8 < 4 else 0 for i in range(cond_actions_length-2)]
                    new_action_tokens = action_tokens*0 + codebook_size
                    for i in range(0, len(actions)):
                        new_action_tokens[:,i,:] += actions[i]
                    gen_tokens = generate_video(cond_tokens, new_action_tokens, st_transformer, context_size)
                    video_img = make_video_plot([orig_tokens, gen_tokens], vqvae, nrow=cond_video_length+cond_actions_length)
                    wandb.log({f"action=alternating": wandb.Image(video_img,file_type="jpg")})
                    make_video([orig_tokens, gen_tokens], "video_alternating.mp4")
                    wandb.log({f"action=alternating": wandb.Video("video_alternating.mp4")})


            if steps % 10 == 0:
                wandb.log({"train/loss": loss.item()})
            steps += 1

