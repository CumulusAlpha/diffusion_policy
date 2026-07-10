from typing import Dict, Optional, Sequence
import copy

import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.common.normalize_util import (
    array_to_stats,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    get_range_normalizer_from_stat,
)
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)


class AbppDataset(BaseImageDataset):
    def __init__(
            self,
            zarr_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            resize_hw: Optional[Sequence[int]] = None,
            seed=42,
            val_ratio=0.0
        ):
        super().__init__()

        self.replay_buffer = ReplayBuffer.create_from_path(zarr_path, mode='r')
        self.n_obs_steps = n_obs_steps
        self.resize_hw = tuple(resize_hw) if resize_hw is not None else None

        key_first_k = dict()
        if n_obs_steps is not None:
            key_first_k['rgb'] = n_obs_steps
            key_first_k['state'] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            keys=['rgb', 'state', 'action'],
            episode_mask=train_mask,
            key_first_k=key_first_k)

        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.seed = seed
        self.val_ratio = val_ratio

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        key_first_k = dict()
        if self.n_obs_steps is not None:
            key_first_k['rgb'] = self.n_obs_steps
            key_first_k['state'] = self.n_obs_steps
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            keys=['rgb', 'state', 'action'],
            episode_mask=~self.train_mask,
            key_first_k=key_first_k)
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer['action'] = self._get_action_normalizer()

        state = self.replay_buffer['state']
        normalizer['robot_qpos'] = get_range_normalizer_from_stat(
            array_to_stats(state[:, :6]))
        normalizer['xhand_qpos'] = get_range_normalizer_from_stat(
            array_to_stats(state[:, 6:18]))

        for camera_idx in range(3):
            normalizer[f'camera_{camera_idx}'] = get_image_range_normalizer()
        return normalizer

    def _get_action_normalizer(self) -> SingleFieldLinearNormalizer:
        action = self.replay_buffer['action']
        eef_pos_normalizer = get_range_normalizer_from_stat(
            array_to_stats(action[:, :3]))
        eef_quat_normalizer = get_identity_normalizer_from_stat(
            array_to_stats(action[:, 3:7]))
        xhand_qpos_normalizer = get_range_normalizer_from_stat(
            array_to_stats(action[:, 7:19]))

        normalizers = [
            eef_pos_normalizer,
            eef_quat_normalizer,
            xhand_qpos_normalizer,
        ]
        scale = np.concatenate([
            x.params_dict['scale'].detach().cpu().numpy()
            for x in normalizers
        ], axis=0).astype(np.float32)
        offset = np.concatenate([
            x.params_dict['offset'].detach().cpu().numpy()
            for x in normalizers
        ], axis=0).astype(np.float32)
        input_stats_dict = dict()
        for key in ['min', 'max', 'mean', 'std']:
            input_stats_dict[key] = np.concatenate([
                x.params_dict['input_stats'][key].detach().cpu().numpy()
                for x in normalizers
            ], axis=0).astype(np.float32)

        return SingleFieldLinearNormalizer.create_manual(
            scale=scale,
            offset=offset,
            input_stats_dict=input_stats_dict)

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'][:])

    def __len__(self):
        return len(self.sampler)

    def _resize_images(self, images: np.ndarray) -> np.ndarray:
        if self.resize_hw is None:
            return images

        height, width = self.resize_hw
        if images.shape[1:3] == (height, width):
            return images

        import cv2
        resized = [
            cv2.resize(x, (width, height), interpolation=cv2.INTER_AREA)
            for x in images
        ]
        return np.stack(resized, axis=0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)
        t_slice = slice(self.n_obs_steps)

        rgb = data['rgb'][t_slice]
        state = data['state'][t_slice].astype(np.float32)

        obs_dict = {
            'robot_qpos': state[:, :6],
            'xhand_qpos': state[:, 6:18],
        }
        for camera_idx in range(3):
            image = self._resize_images(rgb[:, camera_idx])
            obs_dict[f'camera_{camera_idx}'] = np.moveaxis(
                image, -1, 1).astype(np.float32) / 255.

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(data['action'].astype(np.float32))
        }
        return torch_data
