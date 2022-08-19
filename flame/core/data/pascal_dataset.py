import cv2
import torch
import random
import numpy as np

from typing import Dict, Tuple, List, Optional
from pathlib import Path
from torch.utils.data import Dataset
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


class PascalDataset(Dataset):
    def __init__(
        self,
        image_dir: str = None,
        label_dir: str = None,
        txt_path: str = None,
        classes: Dict[str, List[int]] = None,
        image_size: Tuple[int, int] = (224, 224),
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        image_extent: str = ".jpg",
        label_extent: str = ".png",
        transforms: Optional[List] = None,
        require_transforms: Optional[List] = None,
    ) -> None:
        super(PascalDataset, self).__init__()
        self.classes = classes
        self.image_size = image_size
        self.transforms = transforms if transforms else []
        self.require_transforms = require_transforms if require_transforms else []

        self.mean = torch.tensor(mean, dtype=torch.float).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float).view(3, 1, 1)

        image_dir = Path(image_dir)
        label_dir = Path(label_dir)

        with Path(txt_path).open(mode="r", encoding="utf-8") as fp:
            image_names = fp.read().splitlines()

        self.data_pairs = []
        for image_name in image_names:
            image_path = image_dir.joinpath(f"{image_name}{image_extent}")
            label_path = label_dir.joinpath(f"{image_name}{label_extent}")
            if image_path.exists() and label_path.exists():
                self.data_pairs.append((image_path, label_path))

        print(f"{Path(txt_path).stem} - {len(self.data_pairs)}")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        image_path, label_path = self.data_pairs[idx]
        image, mask = cv2.imread(str(image_path)), cv2.imread(str(label_path))
        image = cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB
        )  # convert image from BGR to RGB to map with order of mean, std values
        mask = cv2.cvtColor(
            mask, cv2.COLOR_BGR2RGB
        )  # convert image type of mask from BGR to RGB to map colors pallet

        if image.shape != mask.shape:
            raise ValueError(
                f"{image_path} - image and mask does not have the same shape."
            )

        image_info = [str(image_path), image.shape[1::-1]]
        mask = SegmentationMapsOnImage(mask, image.shape)

        for transform in random.sample(
            self.transforms, k=random.randint(0, len(self.transforms))
        ):
            image, mask = transform(image=image, segmentation_maps=mask)

        for require_transform in self.require_transforms:
            image, mask = require_transform(image=image, segmentation_maps=mask)

        mask = mask.get_arr()

        image = cv2.resize(image, dsize=self.image_size)
        mask = cv2.resize(mask, dsize=self.image_size, interpolation=cv2.INTER_NEAREST)

        colors = np.asarray(list(self.classes.values()))  # [num_classes, 3]
        colors = np.expand_dims(colors, axis=1)  # [num_classes, 1, 3]
        colors = np.expand_dims(colors, axis=1)  # [num_classes, 1, 1, 3]

        mask = (mask == colors).astype(np.uint8)  # [num_classe, H, W, 3]
        mask = mask.prod(axis=-1)  # [num_classes, H, W]
        mask = mask.argmax(axis=0)  # [H, W]

        image, mask = np.ascontiguousarray(image), np.ascontiguousarray(mask)
        image, mask = torch.from_numpy(image), torch.from_numpy(mask)

        image = image.permute(2, 0, 1).contiguous()  # [C, H, W]
        image = (image.float().div(255.0) - self.mean) / self.std

        return image, mask, image_info
