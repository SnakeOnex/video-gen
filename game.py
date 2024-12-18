import torch, torch.nn as nn, torchvision, argparse, time, tqdm, PIL, wandb, torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from vqvae import VQGAN, VQGANConfig
from gpt import GPTLanguageModel, GPTConfig
from dataset import VideoDataset, annotate_video, encode_video, action_to_text
from train_st import pack_tokens, unpack_tokens, generate_video, make_video_plot, TrainerConfig

device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == '__main__':
    gpt_path = Path('gpt_best_context_2.pt')
    context_size = 2


    dataset_path = Path("../teco/dmlab/")
    dataset = VideoDataset(Path("../teco/dmlab/test"))


    vqvae_config = VQGANConfig(
        num_codebook_vectors=512,
        latent_dim=8, 
        resolution=64, 
        ch_mult=(1, 2, 2, 2)
    )

    gpt_config = GPTConfig(block_size=context_size*(vqvae_config.spatial_dim**2+1), 
                           vocab_size=vqvae_config.num_codebook_vectors+3, 
                           n_embd=368, 
                           n_head=4, 
                           n_layer=4, 
                           causal=True, 
                           dropout=0.1)

    config = TrainerConfig(gpt_config=gpt_config,
                           gpt_path=gpt_path,
                           vqvae_config=vqvae_config,
                           vqvae_path="vqvae_best.pt",
                           dataset_path=dataset_path,
                           gen_video_size=12,
                           epochs=100,
                           batch_size=16,
                           lr=3e-4,
                           log_interval=10,
                           eval_interval=1000,
                           vis_count=4,
                           cond_video_length=4,
                           cond_actions_length=10)

    vqvae = VQGAN(vqvae_config).to(device)
    vqvae.load_state_dict(torch.load("vqvae_dmlab_best.pt", weights_only=True, map_location=device))
    gpt_config = GPTConfig(block_size=context_size*(vqvae_config.spatial_dim**2+1), 
                           vocab_size=vqvae_config.num_codebook_vectors+3, 
                           n_embd=368, 
                           n_head=4, 
                           n_layer=4, 
                           causal=True, 
                           dropout=0.1)
    st_transformer = GPTLanguageModel(gpt_config).to(device)
    st_transformer.load_state_dict(torch.load(gpt_path, weights_only=True, map_location=device))
    st_transformer.eval()

    cond_actions_length = 6+1

    # 1. get video
    video, action = dataset[0]
    video = video.unsqueeze(0).to(device)
    action = action.unsqueeze(0).to(device)
    video = video[:,:1]
    action = action[:,:1]
    tokens = pack_tokens(video, action, vqvae)
    cond_tokens = tokens[:,:1,:]
    print(tokens.shape)

    fig, ax = plt.subplots(1, 1)

    def button_press(event):
        global action_tokens, cond_tokens
        print("Button press: ", event.key)
        if event.key == 'a':
            action = 0
        elif event.key == 'd':
            action = 1
        elif event.key == 'w':
            action = 2
        else:
            return
        action_tokens = action_tokens * 0 + action + config.codebook_size
        # cond_tokens = cond_tokens[:,1:,:]

    fig.canvas.mpl_connect('key_press_event', button_press)

    while True:
        print("Waiting for key press")
        action_tokens = tokens[:,-1:,config.spatial_dim**2:config.spatial_dim**2+1]
        print("pre-key: ", action_tokens)
        plt.waitforbuttonpress()
        print("post-key: ", action_tokens)
        # action_tokens = action_tokens * 0 + 2 + codebook_size

        # 2. generate video
        cond_tokens = generate_video(cond_tokens, action_tokens, st_transformer, context_size, config.spatial_dim**2)
        cond_tokens = cond_tokens[:,1:,:]
        img = make_video_plot([cond_tokens], vqvae, nrow=cond_actions_length+1)
        ax.imshow(img)
        plt.draw()




