import torch, numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import imageio

class VideoDataset(Dataset):
    def __init__(self, dataset_path):
        self.video_paths = []
        for folder_path in dataset_path.iterdir():
            for video_path in folder_path.iterdir():
                self.video_paths.append(video_path)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        data = np.load(self.video_paths[idx])
        video, action = data['video'], data['actions']
        video = (torch.from_numpy(video).float() / 255) * 2 - 1
        video = video.permute(0, 3, 1, 2)
        action = torch.from_numpy(action).float()
        return video, action

def visualize_video(video, output_path):
    """
    given T x C x H x W tensor, export video to output_path
    """
    video = video.permute(0, 2, 3, 1).cpu().numpy()
    video = ((video + 1) / 2 * 255).astype(np.uint8)

    writer = imageio.get_writer(output_path, fps=10)
    for frame in video:
        writer.append_data(frame)
    writer.close()

if __name__ == '__main__':
    dm_lab_path = Path('../teco/dmlab/train/')

    dataset = VideoDataset(dm_lab_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    video, _ = next(iter(loader))

    for i, (video, action) in enumerate(loader):
        visualize_video(video[0], f'video_{i}.mp4')
        if i == 5:
            break

