from pathlib import Path
import random
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from datasets.lightning_data_module import LightningDataModule


class MedicalBinaryTransforms:
    def __init__(self, img_size: tuple[int, int], train: bool) -> None:
        self.img_size = img_size
        self.train = train

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        if self.train:
            if random.random() > 0.5:
                angle = 90 * random.randint(0, 3)
                image = F.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
                mask = F.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
                if random.random() > 0.5:
                    image, mask = F.hflip(image), F.hflip(mask)
                else:
                    image, mask = F.vflip(image), F.vflip(mask)
            elif random.random() > 0.5:
                angle = random.uniform(-20.0, 20.0)
                image = F.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
                mask = F.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)

        image = F.resize(image, self.img_size, interpolation=InterpolationMode.BILINEAR)
        mask = F.resize(mask, self.img_size, interpolation=InterpolationMode.NEAREST)
        image = F.to_tensor(image)
        mask = torch.from_numpy(np.array(mask, dtype=np.uint8))
        return image, (mask > 0).to(torch.uint8)


class MedicalBinaryFolderDataset(Dataset):
    """Folder dataset with legacy img/labelcol support and flat-BMP manifests."""

    def __init__(
        self,
        dataset_dir: Union[str, Path],
        img_size: tuple[int, int],
        train: bool,
        source_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.source_dir = Path(source_dir) if source_dir is not None else None
        self.transforms = MedicalBinaryTransforms(img_size=img_size, train=train)
        self.samples: list[tuple[Path, Path]] = []

        manifest = next(
            (
                self.dataset_dir / name
                for name in ("samples.txt", "manifest.txt")
                if (self.dataset_dir / name).is_file()
            ),
            None,
        )
        if manifest is not None:
            if self.source_dir is None:
                raise ValueError(f"{manifest} requires source_dir")
            for raw_name in manifest.read_text(encoding="utf-8").splitlines():
                sample_id = raw_name.strip()
                if sample_id:
                    self.samples.append(self._resolve_flat_pair(sample_id))
        else:
            image_dir = self.dataset_dir / "img"
            mask_dir = self.dataset_dir / "labelcol"
            if not image_dir.is_dir():
                raise FileNotFoundError(f"Image directory not found: {image_dir}")
            if not mask_dir.is_dir():
                raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
            for image_path in sorted(image_dir.iterdir()):
                mask_path = mask_dir / f"{image_path.stem}.png"
                if image_path.is_file() and mask_path.is_file():
                    self.samples.append((image_path, mask_path))

        if not self.samples:
            raise FileNotFoundError(f"No matched image/mask pairs found in {self.dataset_dir}")

    def _resolve_flat_pair(self, raw_name: str) -> tuple[Path, Path]:
        stem = Path(raw_name).stem
        if stem.endswith("_anno"):
            stem = stem[:-5]
        image_candidates = [
            self.source_dir / raw_name,
            self.source_dir / f"{stem}.bmp",
            self.source_dir / f"{stem}.png",
        ]
        mask_candidates = [
            self.source_dir / f"{stem}_anno.bmp",
            self.source_dir / f"{stem}_anno.png",
        ]
        image_path = next((p for p in image_candidates if p.is_file()), None)
        mask_path = next((p for p in mask_candidates if p.is_file()), None)
        if image_path is None or mask_path is None:
            raise FileNotFoundError(f"Could not resolve pair for {raw_name} under {self.source_dir}")
        return image_path, mask_path

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, mask_path = self.samples[index]
        image, mask = self.transforms(
            Image.open(image_path).convert("RGB"),
            Image.open(mask_path).convert("L"),
        )
        return image, {
            "masks": mask.unsqueeze(0).bool(),
            "labels": torch.tensor([0], dtype=torch.long),
            "is_crowd": torch.tensor([False]),
            "image_name": image_path.name,
        }


class MedicalBinarySplitDataset(Dataset):
    """Legacy split dataset retained for non-GlaS projects."""

    def __init__(self, dataset_dir: Union[str, Path], split: str, img_size: tuple[int, int], train: bool) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.image_dir = self.dataset_dir / "images"
        self.mask_dir = self.dataset_dir / "masks"
        self.split_path = self.dataset_dir / f"{split}.txt"
        self.transforms = MedicalBinaryTransforms(img_size=img_size, train=train)
        if not self.image_dir.is_dir() or not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Missing images/masks under {self.dataset_dir}")
        if not self.split_path.is_file():
            raise FileNotFoundError(f"Split file not found: {self.split_path}")
        self.samples: list[tuple[Path, Path]] = []
        for raw_name in self.split_path.read_text(encoding="utf-8").splitlines():
            name = raw_name.strip()
            if name:
                self.samples.append((
                    self._resolve_file(self.image_dir, name),
                    self._resolve_file(self.mask_dir, name),
                ))
        if not self.samples:
            raise FileNotFoundError(f"No matched pairs in {self.split_path}")

    @staticmethod
    def _resolve_file(folder: Path, stem_or_name: str) -> Path:
        direct = folder / stem_or_name
        if direct.is_file():
            return direct
        for ext in (".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG"):
            candidate = folder / f"{stem_or_name}{ext}"
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"Could not resolve {stem_or_name} under {folder}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, mask_path = self.samples[index]
        image, mask = self.transforms(
            Image.open(image_path).convert("RGB"),
            Image.open(mask_path).convert("L"),
        )
        return image, {
            "masks": mask.unsqueeze(0).bool(),
            "labels": torch.tensor([0], dtype=torch.long),
            "is_crowd": torch.tensor([False]),
            "image_name": image_path.name,
        }


class MedicalBinarySemantic(LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        test_dir: Optional[str] = None,
        img_size: tuple[int, int] = (224, 224),
        num_classes: int = 2,
        batch_size: int = 14,
        num_workers: int = 8,
        check_empty_targets: bool = False,
        ignore_idx: int = 255,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        dataset_dir: Optional[str] = None,
        source_dir: Optional[str] = None,
        train_split: str = "train",
        val_split: str = "val",
        test_split: Optional[str] = None,
    ) -> None:
        super().__init__(
            path="",
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
            ignore_idx=ignore_idx,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        self.save_hyperparameters(ignore=["_class_path"])

    def setup(self, stage: Union[str, None] = None):
        use_split_dataset = self.hparams.dataset_dir is not None
        if stage == "fit" or stage is None:
            if use_split_dataset:
                self.train_dataset = MedicalBinarySplitDataset(self.hparams.dataset_dir, self.hparams.train_split, self.img_size, True)
                self.val_dataset = MedicalBinarySplitDataset(self.hparams.dataset_dir, self.hparams.val_split, self.img_size, False)
            else:
                self.train_dataset = MedicalBinaryFolderDataset(self.hparams.train_dir, self.img_size, True, self.hparams.source_dir)
                self.val_dataset = MedicalBinaryFolderDataset(self.hparams.val_dir, self.img_size, False, self.hparams.source_dir)
        if stage == "validate":
            if use_split_dataset:
                self.val_dataset = MedicalBinarySplitDataset(self.hparams.dataset_dir, self.hparams.val_split, self.img_size, False)
            else:
                self.val_dataset = MedicalBinaryFolderDataset(self.hparams.val_dir, self.img_size, False, self.hparams.source_dir)
        if stage == "test" or stage is None:
            if use_split_dataset:
                self.test_dataset = MedicalBinarySplitDataset(
                    self.hparams.dataset_dir,
                    self.hparams.test_split or self.hparams.val_split,
                    self.img_size,
                    False,
                )
            else:
                test_dir = self.hparams.test_dir or self.hparams.val_dir
                self.test_dataset = MedicalBinaryFolderDataset(test_dir, self.img_size, False, self.hparams.source_dir)
        return self

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, drop_last=True, collate_fn=self.train_collate, **self.dataloader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, drop_last=False, collate_fn=self.train_collate, **self.dataloader_kwargs)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            drop_last=False,
            collate_fn=self.train_collate,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.dataloader_kwargs["pin_memory"],
            persistent_workers=self.dataloader_kwargs["persistent_workers"],
        )
