import torch, numpy as np
import cv2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import imageio

def draw_text(img, text):
    cv2.putText(img, text, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

def action_to_text(action):
    return [f"{word}" for word in ["left", "right", "forward"]][action]

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
        action = torch.from_numpy(action)
        return video, action

def annotate_video(video, actions):
    """
    given T x C x H x W tensor, annotates frames with action text and returns as a numpy array
    """
    video = video.permute(0, 2, 3, 1).detach().cpu().numpy()
    video = ((video + 1) / 2 * 255).astype(np.uint8).copy()
    for i in range(video.shape[0]):
        draw_text(video[i], action_to_text(actions[i]))
    return video

def encode_video(video, save_path):
    writer = imageio.get_writer(save_path, fps=4)
    for frame in video:
        writer.append_data(frame)
    writer.close()

if __name__ == '__main__':
    dm_lab_path = Path('../teco/dmlab/train/')

    dataset = VideoDataset(dm_lab_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    video, _ = next(iter(loader))

    for i, (video, action) in enumerate(loader):
        video = annotate_video(video[0], action[0])
        encode_video(video, f"video_{i}.mp4")
        if i == 5:
            break



