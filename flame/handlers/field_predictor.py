import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple
from shapely.geometry import Polygon

from ..module import Module
from ignite.engine import Events


class FieldPredictor(Module):
    def __init__(
        self,
        evaluator_name: str = "test",
        image_extent: str = ".jpg",
        mask_extent: str = ".png",
        classes: dict = None,
        output_dir: str = None,
        mk_field_dir: bool = False,
        threshold: float = None,
        prob_threshold: float = 0.5,
        output_transform=lambda x: x,
    ):
        super(FieldPredictor, self).__init__()
        self.classes = classes
        self.threshold = threshold
        self.mask_extent = mask_extent
        self.image_extent = image_extent
        self.prob_threshold = prob_threshold
        self.evaluator_name = evaluator_name
        self._output_transform = output_transform

        self.mk_field_dir = mk_field_dir
        self.output_dir = Path(output_dir)
        if (not self.mk_field_dir) and (not self.output_dir.exists()):
            self.output_dir.mkdir(parents=True)

    def init(self):
        assert (
            self.evaluator_name in self.frame
        ), f"The frame does not have {self.evaluator_name}"
        self._attach(self.frame[self.evaluator_name].engine)

    def reset(self):
        pass

    def update(self, output):
        preds, image_infos = output

        image_paths, original_sizes = image_infos
        original_sizes = [(w.item(), h.item()) for w, h in zip(*original_sizes)]

        # preds = (preds > self.prob_threshold).to(torch.uint8)  # preds: B x num_classes x H x W
        preds = [
            pred.squeeze(dim=0).cpu().numpy()
            for pred in torch.split(preds, split_size_or_sections=1, dim=0)
        ]

        for pred, image_path, original_size in zip(preds, image_paths, original_sizes):
            if self.mk_field_dir:
                _output_dir = self.output_dir.joinpath(Path(image_path).parent.stem)
                if not _output_dir.exists():
                    _output_dir.mkdir(parents=True)
            else:
                _output_dir = self.output_dir

            save_image_path = _output_dir.joinpath(Path(image_path).name).with_suffix(
                self.image_extent
            )
            save_mask_path = _output_dir.joinpath(
                f"{Path(image_path).stem}_mask{self.mask_extent}"
            )

            thickness = max(original_size) // 500
            image = cv2.imread(image_path)
            color_mask = np.zeros(shape=(*original_size[::-1], 3), dtype=np.uint8)

            for field_name, (
                class_color,
                class_idx,
                is_expand_height,
                is_expand_width,
                class_ratio,
            ) in self.classes.items():
                mask = cv2.resize(pred[class_idx], dsize=original_size)
                # remove tiny predicted region in class mask
                mask = self.rm_small_components(mask, class_ratio)

                # get all connected components in mask
                num_labels, label = cv2.connectedComponents(
                    image=(mask > self.prob_threshold).astype(np.uint8)
                )

                boxes = []
                for i in range(1, num_labels):
                    # get all contours in binary mask (before setting class pred with probability threshold)
                    contours = cv2.findContours(
                        image=(label == i).astype(np.uint8),
                        mode=cv2.RETR_EXTERNAL,
                        method=cv2.CHAIN_APPROX_SIMPLE,
                    )[-2]

                    # draw contours to mask
                    cv2.drawContours(
                        image=color_mask,
                        contours=contours,
                        contourIdx=-1,
                        color=(255, 255, 255),
                        thickness=thickness,
                    )

                    if len(contours) != 1:
                        raise RuntimeError(
                            "Found more than one contour in one connected component."
                        )

                    # get min area rectangle of external contour
                    contour = contours[0]
                    box = cv2.boxPoints(cv2.minAreaRect(contour))

                    if is_expand_height and (Polygon(box).area > 0):
                        box = self.expand_height(box)
                    # # TODO:
                    # elif is_expand_width and (Polygon(box).area > 0):
                    #     box = self.expand_width(box)

                    box = self.order_points(box)
                    boxes.append(box)

                # draw text boxes to image
                cv2.polylines(
                    img=image,
                    pts=np.int32(np.round(boxes)),
                    isClosed=True,
                    color=class_color,
                    thickness=thickness,
                )

                # draw text colors to mask
                color_mask[label.astype(bool)] = np.array(class_color, dtype=np.uint8)

            cv2.imwrite(str(save_image_path), image)
            cv2.imwrite(str(save_mask_path), color_mask)

    def rm_small_components(self, mask: np.ndarray, class_ratio: float) -> np.ndarray:
        num_class, label = cv2.connectedComponents(mask.round().astype(np.uint8))
        threshold = self.threshold * class_ratio * mask.shape[0] * mask.shape[1]

        for i in range(1, num_class):
            area = (label == i).sum()
            if area < threshold:
                mask[label == i] = 0

        return mask

    def expand_height(self, points: np.ndarray) -> np.ndarray:
        if self.distance(points[0], points[1]) > self.distance(points[0], points[3]):
            points[0] = points[0] - 0.50 * (points[3] - points[0])
            points[1] = points[1] - 0.50 * (points[2] - points[1])
            points[3] = points[0] + 4 / 3 * (points[3] - points[0])
            points[2] = points[1] + 4 / 3 * (points[2] - points[1])
        else:
            points[0] = points[0] - 0.50 * (points[1] - points[0])
            points[3] = points[3] - 0.50 * (points[2] - points[3])
            points[1] = points[0] + 4 / 3 * (points[1] - points[0])
            points[2] = points[3] + 4 / 3 * (points[2] - points[3])
        return points

    def distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        return np.linalg.norm(point1 - point2).item()

    def order_points(
        self, points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        assert len(points) == 4, "Length of points must be 4"
        ls = sorted(points, key=lambda p: p[0])[:2]  # two points at left side
        rs = sorted(points, key=lambda p: p[0])[2:]  # two points at right side
        tl, bl = sorted(
            ls, key=lambda p: p[1]
        )  # top point and bottom point in left side
        tr, br = sorted(
            rs, key=lambda p: p[1]
        )  # top point and bottom point in right side
        return [tl, tr, br, bl]

    def compute(self):
        pass

    def started(self, engine):
        self.reset()

    @torch.no_grad()
    def iteration_completed(self, engine):
        output = self._output_transform(engine.state.output)
        self.update(output)

    def completed(self, engine):
        self.compute()

    def _attach(self, engine):
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.completed)
        if not engine.has_event_handler(self.started, Events.EPOCH_STARTED):
            engine.add_event_handler(Events.EPOCH_STARTED, self.started)
        if not engine.has_event_handler(
            self.iteration_completed, Events.ITERATION_COMPLETED
        ):
            engine.add_event_handler(
                Events.ITERATION_COMPLETED, self.iteration_completed
            )
