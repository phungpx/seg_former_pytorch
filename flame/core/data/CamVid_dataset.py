import cv2
import torch
import random
import numpy as np

from pathlib import Path
from natsort import natsorted
from torch.utils.data import Dataset
from typing import Dict, Tuple, List, Optional
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


class CamVidDataset(Dataset):
    def __init__(
        self,
        image_dir: str = "dataset/CamVid/train/",
        mask_dir: str = "dataset/CamVid/trainannot",
        classes: Dict[int, list] = {
            "sky": [[229, 255, 0], 0],
            "building": [[0, 179, 255], 1],
            "pole": [[0, 255, 208], 2],
            "road": [[252, 3, 219], 3],
            "pavement": [[252, 3, 69], 4],
            "tree": [[0, 255, 0], 5],
            "signsymbol": [[0, 0, 255], 6],
            "fence": [[0, 0, 255], 7],
            "car": [[255, 0, 0], 8],
            "pedestrian": [[128, 192, 0], 9],
            "bicyclist": [[0, 128, 128], 10],
            "unlabelled": [[128, 3, 69], 11],
        },
        imsize: Tuple[int, int] = (224, 224),
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        image_extent: str = ".png",
        mask_extent: str = ".png",
        transforms: Optional[List] = None,
        require_transforms: Optional[List] = None,
        save_mask_dir: Optional[str] = None,  # path to save visualized mask
    ) -> None:
        super(CamVidDataset, self).__init__()
        self.imsize = imsize
        self.classes = classes
        self.save_mask_dir = save_mask_dir
        self.transforms = transforms if transforms else []
        self.require_transforms = require_transforms if require_transforms else []

        self.mean = torch.tensor(mean, dtype=torch.float).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float).view(3, 1, 1)

        image_paths = natsorted(
            Path(image_dir).glob(f"*{image_extent}"), key=lambda x: x.stem
        )
        mask_paths = natsorted(
            Path(mask_dir).glob(f"*{mask_extent}"), key=lambda x: x.stem
        )
        self.data_pairs = [
            (image_path, mask_path)
            for image_path, mask_path in zip(image_paths, mask_paths)
            if image_path.stem == mask_path.stem
        ]

        print(f"{Path(image_dir).stem} - {len(self.data_pairs)}")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        """
        Output:
            image: torch.FloatTensor, C X H X W
            mask: torch.Uint8Tensor, num_classes x H x W
            image_info: List[image_path, Tuple(image_width, image_height)]
        """
        image_path, mask_path = self.data_pairs[idx]

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB
        )  # convert image from BGR to RGB to map with order of mean, std values
        mask = cv2.imread(str(mask_path), 0)  # read image in grayscale mode

        # visualize mask
        if self.save_mask_dir is not None:
            save_dir = Path(self.save_mask_dir).joinpath(f"{image_path.parent.stem}seg")
            if not save_dir.exists():
                save_dir.mkdir(parents=True)

            color_mask = np.zeros_like(image)
            for class_name, (class_color, class_id) in self.classes.items():
                object_mask = np.zeros_like(image)
                color_mask[mask == class_id] = np.array(class_color, dtype=np.uint8)
                object_mask[mask == class_id] = np.array(class_color, dtype=np.uint8)
                cv2.imwrite(
                    str(save_dir.joinpath(f"{mask_path.stem}_{class_name}.png")),
                    object_mask,
                )

            cv2.imwrite(str(save_dir.joinpath(f"{mask_path.stem}.png")), color_mask)

        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"`{image_path}` - image and mask does not have the same shape."
            )

        if len(mask.shape) != 2:
            raise ValueError(
                f"`{mask_path}` - the number of dimension of mask is not equal 2."
            )

        image_info = [str(image_path), image.shape[1::-1]]

        # apply augmentation to mask and image
        mask = SegmentationMapsOnImage(mask, image.shape[:2])

        for transform in random.sample(
            self.transforms, k=random.randint(0, len(self.transforms))
        ):
            image, mask = transform(image=image, segmentation_maps=mask)

        for require_transform in self.require_transforms:
            image, mask = require_transform(image=image, segmentation_maps=mask)

        mask = mask.get_arr()

        # resize image, mask to input size of model
        image = cv2.resize(image, dsize=self.imsize)
        mask = cv2.resize(
            mask, dsize=self.imsize, interpolation=cv2.INTER_NEAREST
        )  # !important to set interpolation mode for mask is INTER_NEAREST

        # # convert mask [H, W] to nd array [num_classes, H, W]
        # masks = [(mask == class_id).astype(np.uint8) for class_name, (class_color, class_id) in self.classes.items()]
        # mask = np.stack(masks, axis=0)  # [num_classes, H, W]
        # mask = mask.argmax(axis=0)

        # convert ndarray to torch tensor
        image = torch.from_numpy(np.ascontiguousarray(image))
        mask = torch.from_numpy(np.ascontiguousarray(mask))

        # normalize image
        image = image.permute(2, 0, 1).contiguous()  # [C, H, W]
        image = (image.float().div(255.0) - self.mean) / self.std

        return image, mask.long(), image_info
