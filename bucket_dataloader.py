"""
Aspect Ratio Bucketing DataLoader for Text-to-Image Training (Single GPU)
=========================================================================
Supports square / portrait / landscape images with near-identical pixel counts.
Each batch contains images of the same bucket → zero padding waste.

Usage:
    dataset = T2IDataset("path/to/data", tokenizer=tokenizer)
    loader  = build_dataloader(dataset, batch_size=4)

    for batch in loader:
        pixel_values = batch["pixel_values"]   # (B, 3, H, W)
        input_ids    = batch["input_ids"]       # (B, seq_len)
        ...
"""

from __future__ import annotations

import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Sampler

from typing import Union
try:
    from datasets import Dataset as HFDataset
    HF_AVAILABLE = True
except ImportError:
    HFDataset = None
    HF_AVAILABLE = False

# ---------------------------------------------------------------------------
# 1.  Bucket definitions
# ---------------------------------------------------------------------------

def generate_buckets(
    min_px: int = 512,
    max_px: int = 1024,
    step: int = 64,
    max_area: int = 512 * 512,
) -> list[tuple[int, int]]:
    """
    Generate all (W, H) pairs where:
      - W and H are multiples of `step`  (VAE / attention-tile friendly)
      - min_px <= W, H <= max_px
      - W * H <= max_area
    Returns deduplicated, sorted list of (W, H) tuples.
    """
    buckets: set[tuple[int, int]] = set()
    for w in range(min_px, max_px + 1, step):
        for h in range(min_px, max_px + 1, step):
            if w * h <= max_area:
                buckets.add((w, h))
    return sorted(buckets)


# Sane defaults used throughout; override by passing `buckets=` to the sampler.
#
# max_area controls total pixel budget per image (keeps VRAM predictable).
# min_px / max_px define the side-length range; step=64 keeps dimensions
# divisible by the typical VAE downscale factor (8) and attention tile (8).
#
# Preset options:
#   ~512px target  → generate_buckets(384, 768,  step=64, max_area=512*512)
#   ~768px target  → generate_buckets(512, 1024, step=64, max_area=768*768)   ← default
#   ~1024px target → generate_buckets(512, 1280, step=64, max_area=1024*1024)
DEFAULT_BUCKETS: list[tuple[int, int]] = generate_buckets(
    min_px=512, max_px=1024, step=64, max_area=768 * 768
)


def assign_bucket(
    img_w: int,
    img_h: int,
    buckets: list[tuple[int, int]] = DEFAULT_BUCKETS,
) -> tuple[int, int]:
    """
    Return the bucket (W, H) that best matches the image's aspect ratio
    while staying within the pixel budget.

    Scoring: minimise |log(img_ar) - log(bucket_ar)| so that e.g. 3:4 and
    4:3 are not confused even when their pixel distances are similar.
    """
    img_ar = img_w / img_h
    best_bucket = min(
        buckets,
        key=lambda b: abs(math.log(img_ar) - math.log(b[0] / b[1])),
    )
    return best_bucket


# ---------------------------------------------------------------------------
# 2.  Per-bucket transforms
# ---------------------------------------------------------------------------

def build_transform(bucket_w: int, bucket_h: int) -> transforms.Compose:
    """
    Returns a deterministic transform pipeline for a specific (W, H) bucket:
      1. Resize the shorter side so the image covers the bucket (no black bars)
      2. CenterCrop to exactly (H, W)
      3. Normalise to [-1, 1]  (standard for diffusion models)
    """
    return transforms.Compose([
        # transforms.Resize(
        #     # Resize shorter side to max(W, H) so crop never up-samples.
        #     max(bucket_w, bucket_h),
        #     interpolation=transforms.InterpolationMode.LANCZOS,
        # ),
        transforms.CenterCrop((bucket_h, bucket_w)),   # (H, W) order for PIL
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


# ---------------------------------------------------------------------------
# 3.  Dataset
# ---------------------------------------------------------------------------

class T2IDataset(Dataset):
    """
    Text-to-Image dataset that reads (image, caption) pairs.

    Expected directory layout:
        root/
          images/
            000001.jpg
            000002.png
            ...
          captions.json        ← {"000001.jpg": "a cat on a mat", ...}
                               OR each image has a sibling .txt file

    Args:
        root:        Path to dataset root.
        tokenizer:   Any HuggingFace-compatible tokenizer (optional).
                     If None, `input_ids` will be absent from the batch.
        buckets:     List of (W, H) resolution buckets.
        max_length:  Tokenizer max sequence length.
        flip_prob:   Probability of random horizontal flip (augmentation).
    """

    def __init__(
        self,
        # root: str | Path,
        source: Union[str, Path, "HFDataset"],
        tokenizer: Any = None,
        buckets: list[tuple[int, int]] = DEFAULT_BUCKETS,
        max_length: int = 77,
        flip_prob: float = 0.5,
        image_column: str = "image",
        caption_column: str = "text",
    ) -> None:
        # self.root = Path(root)
        self.tokenizer = tokenizer
        self.buckets = buckets
        self.max_length = max_length
        self.flip_prob = flip_prob
        self.image_column = image_column
        self.caption_column = caption_column

        # self.samples: list[dict] = self._load_samples()
        if HF_AVAILABLE and isinstance(source, HFDataset):
            self.root = None
            self.hf_dataset = source
            self.samples: list[dict] = self._load_samples_from_hf()
        else:
            self.root = Path(source)
            self.hf_dataset = None
            self.samples: list[dict] = self._load_samples()   # existing method unchanged

        self._transform_cache: dict[tuple[int, int], transforms.Compose] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_samples(self) -> list[dict]:
        img_dir = self.root / "images"
        assert img_dir.exists(), f"Image directory not found: {img_dir}"

        # --- captions.json (preferred) ---
        caption_file = self.root / "captions.json"
        if caption_file.exists():
            with open(caption_file) as f:
                captions: dict[str, str] = json.load(f)
        else:
            captions = {}

        samples = []
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue

            # Caption: JSON lookup → sibling .txt → empty string
            caption = captions.get(img_path.name, "")
            if not caption:
                txt_path = img_path.with_suffix(".txt")
                if txt_path.exists():
                    caption = txt_path.read_text().strip()

            # Read image dimensions *without* decoding full pixel data.
            try:
                with Image.open(img_path) as im:
                    w, h = im.size          # PIL: (width, height)
            except Exception:
                continue                   # skip corrupt files

            bucket = assign_bucket(w, h, self.buckets)
            samples.append({
                "img_path": img_path,
                "caption": caption,
                "orig_size": (w, h),
                "bucket": bucket,
            })

        assert samples, f"No valid images found under {img_dir}"
        return samples

    def _load_samples_from_hf(self) -> list[dict]:
        """
        Build the same self.samples structure from a HuggingFace Dataset.

        Handles three image column formats that appear on the Hub:
          1. PIL.Image already decoded  (most datasets with datasets.Image() feature)
          2. dict {"path": str, "bytes": bytes}  (streaming / uncast datasets)
          3. str path  (rare, some older datasets)
        """
        assert self.image_column in self.hf_dataset.column_names, (
            f"image_column='{self.image_column}' not found. "
            f"Available columns: {self.hf_dataset.column_names}"
        )
        assert self.caption_column in self.hf_dataset.column_names, (
            f"caption_column='{self.caption_column}' not found. "
            f"Available columns: {self.hf_dataset.column_names}"
        )

        samples = []
        for row in self.hf_dataset:
            img_raw = row[self.image_column]
            caption = row.get(self.caption_column, "") or ""

            # Normalise to PIL.Image
            if isinstance(img_raw, Image.Image):
                img = img_raw
            elif isinstance(img_raw, dict):
                # {"path": ..., "bytes": ...}  — bytes takes priority
                if img_raw.get("bytes"):
                    import io
                    img = Image.open(io.BytesIO(img_raw["bytes"]))
                else:
                    img = Image.open(img_raw["path"])
            elif isinstance(img_raw, str):
                img = Image.open(img_raw)
            else:
                continue   # skip unrecognisable format

            w, h = img.size
            bucket = assign_bucket(w, h, self.buckets)
            samples.append({
                "img": img,            # ← PIL stored directly (no path to re-open)
                "img_path": None,
                "caption": caption,
                "orig_size": (w, h),
                "bucket": bucket,
            })

        assert samples, "No valid samples loaded from HF dataset."
        return samples

    def _get_transform(self, bucket: tuple[int, int]) -> transforms.Compose:
        if bucket not in self._transform_cache:
            self._transform_cache[bucket] = build_transform(*bucket)
        return self._transform_cache[bucket]

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    @property
    def sizes(self) -> list[tuple[int, int]]:
        """List of (W, H) for every sample — used by the sampler."""
        return [s["orig_size"] for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        bucket_w, bucket_h = sample["bucket"]
        tf = self._get_transform(sample["bucket"])

        # Load and transform image
        # img = Image.open(sample["img_path"]).convert("RGB")
        # HF path: image already in memory; disk path: load from file
        if sample.get("img") is not None:
            img = sample["img"].convert("RGB")
        else:
            img = Image.open(sample["img_path"]).convert("RGB")

        if random.random() < self.flip_prob:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        pixel_values = tf(img)                 # (3, H, W)  in [-1, 1]

        out: dict[str, Any] = {
            "pixel_values": pixel_values,
            "caption": sample["caption"],
            "bucket": sample["bucket"],
            "orig_size": sample["orig_size"],
        }

        # Tokenise caption if a tokenizer is provided
        if self.tokenizer is not None:
            tokens = self.tokenizer(
                sample["caption"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            out["input_ids"] = tokens["input_ids"].squeeze(0)
            out["attention_mask"] = tokens["attention_mask"].squeeze(0)

        return out


# ---------------------------------------------------------------------------
# 4.  Aspect-Ratio Bucket Sampler
# ---------------------------------------------------------------------------

class AspectRatioBucketSampler(Sampler):
    """
    Yields *batch-sized index lists* where every index in a batch belongs to
    the same resolution bucket.

    Features
    --------
    - Shuffle within each bucket every epoch (set seed for reproducibility).
    - Drop the last incomplete batch per bucket (keeps shapes uniform).
    - Shuffle the batch order globally so bucket ordering is random.

    Args:
        dataset:    A T2IDataset instance (needs `.sizes` and `.samples`).
        batch_size: Images per batch.
        shuffle:    Randomise order each epoch.
        seed:       Base RNG seed; actual seed = seed + epoch.
    """

    def __init__(
        self,
        dataset: T2IDataset,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Build bucket → [idx, ...] mapping once
        self.bucket_to_indices: dict[tuple[int, int], list[int]] = defaultdict(list)
        for idx, sample in enumerate(dataset.samples):
            self.bucket_to_indices[sample["bucket"]].append(idx)

        self._log_bucket_stats()

    def _log_bucket_stats(self) -> None:
        total = sum(len(v) for v in self.bucket_to_indices.values())
        print(f"[Sampler] {len(self.bucket_to_indices)} active buckets, "
              f"{total} total samples")
        for bucket, indices in sorted(self.bucket_to_indices.items()):
            n_batches = len(indices) // self.batch_size
            print(f"  bucket {bucket[0]}×{bucket[1]}: "
                  f"{len(indices)} samples → {n_batches} batches")

    def set_epoch(self, epoch: int) -> None:
        """Call before each epoch to get different shuffles."""
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)

        all_batches: list[list[int]] = []
        for bucket_key in sorted(self.bucket_to_indices.keys()):
            indices = self.bucket_to_indices[bucket_key].copy()
            if self.shuffle:
                rng.shuffle(indices)
            # Drop the last incomplete batch
            for start in range(0, len(indices) - self.batch_size + 1, self.batch_size):
                all_batches.append(indices[start : start + self.batch_size])

        if self.shuffle:
            rng.shuffle(all_batches)

        for batch in all_batches:
            yield batch

    def __len__(self) -> int:
        return sum(
            len(v) // self.batch_size
            for v in self.bucket_to_indices.values()
        )


# ---------------------------------------------------------------------------
# 5.  Collate function
# ---------------------------------------------------------------------------

def collate_fn(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Stack a list of same-bucket samples into a batch.

    All pixel tensors in the list share shape (3, H, W) because the sampler
    guarantees same-bucket batches, so torch.stack works without padding.
    """
    batch: dict[str, Any] = {}

    # --- pixel values: (B, 3, H, W) ---
    pixel_values = torch.stack([s["pixel_values"] for s in samples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    batch["pixel_values"] = pixel_values

    # --- captions: kept as list[str] ---
    batch["captions"] = [s["caption"] for s in samples]

    # --- optional token tensors ---
    if "input_ids" in samples[0]:
        batch["input_ids"] = torch.stack([s["input_ids"] for s in samples])
        # batch["attention_mask"] = torch.stack(
        #     [s["attention_mask"] for s in samples]
        # )

    # --- metadata (useful for logging / debugging) ---
    batch["bucket"] = samples[0]["bucket"]          # all identical in batch
    batch["orig_sizes"] = [s["orig_size"] for s in samples]

    return batch


# ---------------------------------------------------------------------------
# 6.  Public factory
# ---------------------------------------------------------------------------

def from_hub(
    dataset_name: str,
    split: str = "train",
    image_column: str = "image",
    caption_column: str = "text",
    tokenizer: Any = None,
    text_tokens_max_length: int = 75,
    buckets: list[tuple[int, int]] = DEFAULT_BUCKETS,
    hf_kwargs: dict = {},
) -> T2IDataset:
    """
    One-liner to load any image-caption dataset from the HuggingFace Hub.

    Examples:
        ds = from_hub("lambdalabs/pokemon-blip-captions",
                      caption_column="text")

        ds = from_hub("laion/laion-art",
                      caption_column="TEXT",
                      hf_kwargs={"streaming": True})  # for huge datasets

        ds = from_hub("nlphuji/flickr30k",
                      caption_column="caption")
    """
    from datasets import load_dataset
    hf_ds = load_dataset(dataset_name, split=split, **hf_kwargs)
    return T2IDataset(
        source=hf_ds,
        tokenizer=tokenizer,
        max_length=text_tokens_max_length,
        buckets=buckets,
        image_column=image_column,
        caption_column=caption_column,
    )

def build_dataloader(
    dataset: T2IDataset,
    batch_size: int = 4,
    num_workers: int = 2,
    shuffle: bool = True,
    seed: int = 42,
    pin_memory: bool = True,
) -> tuple[DataLoader, AspectRatioBucketSampler]:
    """
    Build a DataLoader + sampler pair ready for single-GPU training.

    Returns both so the training loop can call `sampler.set_epoch(epoch)`.

    Example
    -------
        loader, sampler = build_dataloader(dataset, batch_size=4)
        for epoch in range(num_epochs):
            sampler.set_epoch(epoch)
            for batch in loader:
                ...
    """
    sampler = AspectRatioBucketSampler(
        dataset, batch_size=batch_size, shuffle=shuffle, seed=seed
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,          # batch_sampler overrides batch_size/shuffle
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return loader, sampler
