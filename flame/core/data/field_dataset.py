import cv2
import json
import torch
import random
import numpy as np

from typing import List, Tuple, Dict, Optional
from pathlib import Path
from natsort import natsorted
from torch.utils.data import Dataset
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


class FieldDataset(Dataset):
    def __init__(
        self,
        dirnames: List[str],
        classes: Dict[str, List[int]],
        image_size: Tuple[int, int] = (224, 224),
        num_transforms: int = 1,
        mean: Optional[Tuple[float, float, float]] = None,
        std: Optional[Tuple[float, float, float]] = None,
        reduce_ratios: Optional[Tuple[float, float]] = None,
        image_extents: List[str] = [".jpg"],
        label_extent: str = ".json",
        transforms: Optional[List] = None,
        require_transforms: Optional[List] = None,
        save_mask: bool = False,
    ) -> None:
        super(FieldDataset, self).__init__()
        self.classes = classes
        self.save_mask = save_mask
        self.image_size = image_size
        self.reduce_ratios = reduce_ratios
        self.transforms = transforms if transforms else []
        self.require_transforms = require_transforms if require_transforms else []
        self.num_transforms = num_transforms if len(self.transforms) > num_transforms and num_transforms \
                                             else len(self.transforms)
        self.mean = (
            torch.tensor(mean, dtype=torch.float).view(3, 1, 1)
            if mean is not None
            else None
        )
        self.std = (
            torch.tensor(std, dtype=torch.float).view(3, 1, 1)
            if std is not None
            else None
        )

        image_paths, label_paths = [], []

        for dirname in dirnames:
            for image_extent in image_extents:
                image_paths.extend(
                    list(Path(dirname).glob("*{}".format(image_extent)))
                )

            label_paths.extend(list(Path(dirname).glob("*{}".format(label_extent))))

        image_paths = natsorted(image_paths, key=lambda x: x.stem)
        label_paths = natsorted(label_paths, key=lambda x: x.stem)

        self.data_pairs = [
            (image, label)
            for image, label in zip(image_paths, label_paths)
            if image.stem == label.stem
        ]

        # print(f"{', '.join([Path(dirname).stem for dirname in dirnames])} - {len(self.data_pairs)}")
        print(f"{Path(dirnames[0]).parent.stem} - {len(self.data_pairs)}")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        image_path, label_path = self.data_pairs[idx]
        image, mask = self.generate_segmap(image_path, label_path)

        if image.shape != mask.shape:
            raise ValueError(
                f"file path {image_path}: image and mask does not have the same shape."
            )

        # save generated mask
        if self.save_mask:
            alpha = 0.6
            stem1 = image_path.parents[0].stem
            stem2 = image_path.parents[1].stem

            output_dir = Path("masks").joinpath(stem2).joinpath(stem1)
            if not output_dir.exists():
                output_dir.mkdir(parents=True)

            combined_mask = (alpha * image + (1.0 - alpha) * mask).astype(np.uint8)
            cv2.imwrite(
                str(output_dir.joinpath(image_path.stem + ".png")), combined_mask
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
        image = torch.from_numpy(np.ascontiguousarray(image))
        mask = torch.from_numpy(np.ascontiguousarray(mask))  # H x W x 3, uint8

        label = torch.zeros(size=self.image_size, dtype=torch.int64)
        for class_name, (color, class_id, _, _) in self.classes.items():
            color = torch.tensor(color, dtype=torch.uint8).view(1, 1, 3)
            label[(mask == color).to(torch.uint8).prod(dim=-1) == 1] = class_id

        image = image.permute(2, 0, 1).float()
        if (self.mean is not None) and (self.std is not None):
            image = (image.div(255.0) - self.mean) / self.std
        else:
            image = (image - image.mean()) / image.std()
        return image, label, image_info

    def generate_segmap(
        self, image_path: Path, label_path: Path
    ) -> Tuple[np.ndarray, np.ndarray]:
        with label_path.open(mode="r") as f:
            json_info = json.load(f)

        image = cv2.imread(str(image_path))
        mask = np.zeros_like(image)

        for shape in json_info["shapes"]:
            if shape["label"] in self.classes:
                color, class_id, is_reduce_height, is_reduce_width = self.classes[
                    shape["label"]
                ]
                if shape["shape_type"] == "polygon":
                    points = shape["points"]
                    if (len(points) != 4) and (is_reduce_height or is_reduce_width):
                        print(f"Error: {image_path} - {shape['label']}")
                elif shape["shape_type"] == "rectangle":
                    points = self.to_4points(shape["points"])
                else:
                    raise ValueError(
                        f"file name {label_path.name}: shape type of region must be rectangle or polygon."
                    )

                if is_reduce_height:
                    points = self.reduce_height(
                        np.array(self.order_points(points)),
                        factor1=self.reduce_ratios[0],
                        factor2=self.reduce_ratios[1],
                    )

                if is_reduce_width:
                    points = self.reduce_width(
                        np.array(self.order_points(points)),
                        factor1=self.reduce_ratios[0],
                        factor2=self.reduce_ratios[1],
                    )

                cv2.fillPoly(img=mask, pts=[np.int32(points)], color=color)
        return image, mask

    def reduce_height(
        self, points: np.ndarray, factor1: float = 0.25, factor2: float = 0.25
    ) -> List[List[int]]:
        assert (
            type(points) == np.ndarray
        ), f"type of points must be np.ndarray instead of {points.dtype}"
        reduced_points = np.zeros_like(points)
        reduced_points[0] = self.point_on_segment(
            points[0], points[3], factor1 * self.distance(points[0], points[3])
        )
        reduced_points[1] = self.point_on_segment(
            points[1], points[2], factor1 * self.distance(points[1], points[2])
        )
        reduced_points[2] = self.point_on_segment(
            points[2], points[1], factor2 * self.distance(points[1], points[2])
        )
        reduced_points[3] = self.point_on_segment(
            points[3], points[0], factor2 * self.distance(points[0], points[3])
        )
        return reduced_points.round().astype(np.int32).tolist()

    def reduce_width(
        self, points: np.ndarray, factor1: float = 0.25, factor2: float = 0.25
    ) -> List[List[int]]:
        assert (
            type(points) == np.ndarray
        ), f"type of points must be np.ndarray instead of {points.dtype}"
        reduced_points = np.zeros_like(points)
        reduced_points[0] = self.point_on_segment(
            points[0], points[1], factor1 * self.distance(points[0], points[3])
        )
        reduced_points[1] = self.point_on_segment(
            points[1], points[0], factor2 * self.distance(points[1], points[2])
        )
        reduced_points[2] = self.point_on_segment(
            points[2], points[3], factor2 * self.distance(points[1], points[2])
        )
        reduced_points[3] = self.point_on_segment(
            points[3], points[2], factor1 * self.distance(points[0], points[3])
        )
        return reduced_points.round().astype(np.int32).tolist()

    def point_on_segment(
        self, point1: np.ndarray, point2: np.ndarray, length_from_point1: float
    ) -> np.ndarray:
        alpha = length_from_point1 / self.distance(point1, point2)
        point = alpha * point2 + (1 - alpha) * point1
        return point

    def distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        return np.linalg.norm(point1 - point2).item()

    def order_points(
        self, points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        assert len(points) == 4, "Length of points must be 4"
        tl = min(points, key=lambda p: p[0] + p[1])
        br = max(points, key=lambda p: p[0] + p[1])
        tr = max(points, key=lambda p: p[0] - p[1])
        bl = min(points, key=lambda p: p[0] - p[1])
        return [tl, tr, br, bl]

    # def order_points(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    #     assert len(points) == 4, 'Length of points must be 4'
    #     ls = sorted(points, key=lambda p: p[0])[:2]  # two points at left side
    #     rs = sorted(points, key=lambda p: p[0])[2:]  # two points at right side
    #     tl, bl = sorted(ls, key=lambda p: p[1])  # top point and bottom point in left side
    #     tr, br = sorted(rs, key=lambda p: p[1])  # top point and bottom point in right side
    #     return [tl, tr, br, bl]

    def to_4points(
        self, points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        x1, y1 = points[0][0], points[0][1]
        x2, y2 = points[1][0], points[1][1]
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
