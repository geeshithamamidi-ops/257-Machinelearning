"""
Stratified device-aware subsampler for PRNU training splits.

Reduces the size of a training split to a target fraction (typically 10-30%)
while preserving every device class and enforcing a hard minimum number of
samples per class so rare devices are never wiped out.

The class labels returned by the existing data loaders may be either strings
(Dresden device ids) or integers (IEEE SP class indices); this sampler treats
the second tuple element opaquely and works with either.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

Sample = tuple[str, Any]


class StratifiedDeviceSampler:
    """
    Reduce a list of ``(path, device_id)`` samples to a stratified subset.

    Guarantees
    ----------
    - Every device class present in the input is represented in the output.
    - Each device is subsampled independently with the same target fraction,
      preserving per-class proportions.
    - A hard floor of ``min_per_class`` is enforced so rare devices survive;
      devices that have fewer than ``min_per_class`` images to begin with
      are kept in full (the effective floor is
      ``min(min_per_class, n_original)``).
    - Reproducible across runs via the provided ``seed``.
    """

    def __init__(
        self,
        fraction: float = 0.20,
        min_per_class: int = 30,
        seed: int = 42,
    ) -> None:
        """
        Parameters
        ----------
        fraction : float
            Target keep fraction, typically in ``[0.10, 0.30]``. Must be in
            ``(0.0, 1.0]``.
        min_per_class : int
            Hard minimum number of samples retained per device, capped at the
            number actually available for that device.
        seed : int
            RNG seed for reproducible sampling.
        """
        if not 0.0 < fraction <= 1.0:
            raise ValueError(
                f"fraction must be in (0.0, 1.0], got {fraction!r}"
            )
        if min_per_class < 1:
            raise ValueError(
                f"min_per_class must be >= 1, got {min_per_class!r}"
            )
        self.fraction = float(fraction)
        self.min_per_class = int(min_per_class)
        self.seed = int(seed)
        self._summary: dict[str, Any] | None = None

    def sample(self, samples: list[Sample]) -> list[Sample]:
        """
        Return a stratified subset of ``samples``.

        Parameters
        ----------
        samples : list[tuple[str, Any]]
            Full split, each tuple is ``(image_path, device_id)``. The second
            element may be either a string or an integer label.

        Returns
        -------
        list[tuple[str, Any]]
            Reduced list, shuffled with the configured seed.

        Raises
        ------
        ValueError
            If any post-sampling invariant is violated (see ``_verify``).
        """
        if not samples:
            self._summary = {
                "total_original": 0,
                "total_sampled": 0,
                "fraction_achieved": 0.0,
                "min_per_class": 0,
                "max_per_class": 0,
                "classes_total": 0,
                "classes_below_floor": 0,
                "per_class_kept": {},
                "per_class_original": {},
            }
            return []

        by_cls: dict[Any, list[Sample]] = defaultdict(list)
        for s in samples:
            by_cls[s[1]].append(s)

        rng = random.Random(self.seed)
        sampled: list[Sample] = []
        per_class_kept: dict[Any, int] = {}
        classes_below_floor = 0

        for cls in sorted(by_cls.keys(), key=lambda k: str(k)):
            items = by_cls[cls]
            n = len(items)
            target = int(n * self.fraction)
            effective_floor = min(self.min_per_class, n)
            n_keep = max(effective_floor, target)
            n_keep = min(n_keep, n)
            picks = rng.sample(items, n_keep)
            sampled.extend(picks)
            per_class_kept[cls] = n_keep
            if n < self.min_per_class:
                classes_below_floor += 1

        rng.shuffle(sampled)

        per_class_original = {c: len(v) for c, v in by_cls.items()}
        total_original = len(samples)
        total_sampled = len(sampled)
        kept_counts = list(per_class_kept.values())
        self._summary = {
            "total_original": total_original,
            "total_sampled": total_sampled,
            "fraction_achieved": total_sampled / total_original,
            "min_per_class": min(kept_counts),
            "max_per_class": max(kept_counts),
            "classes_total": len(per_class_kept),
            "classes_below_floor": classes_below_floor,
            "per_class_kept": per_class_kept,
            "per_class_original": per_class_original,
        }

        self._verify(by_cls, per_class_kept, sampled)
        return sampled

    def _verify(
        self,
        by_cls: dict[Any, list[Sample]],
        per_class_kept: dict[Any, int],
        sampled: list[Sample],
    ) -> None:
        """
        Assert post-sampling invariants.

        Raises
        ------
        ValueError
            If any device disappears, any class has fewer than its effective
            floor, or the total size drifts beyond 5% of the expected target
            (where the expected target already accounts for floor
            adjustments).
        """
        original_classes = set(by_cls.keys())
        sampled_classes = {s[1] for s in sampled}
        missing = original_classes - sampled_classes
        if missing:
            raise ValueError(
                "StratifiedDeviceSampler dropped devices that were present "
                f"in the input split: {sorted(missing, key=str)!r}"
            )

        for cls, kept in per_class_kept.items():
            effective_floor = min(self.min_per_class, len(by_cls[cls]))
            if kept < effective_floor:
                raise ValueError(
                    f"Device {cls!r}: kept {kept} samples but effective "
                    f"floor is {effective_floor} "
                    f"(min_per_class={self.min_per_class}, "
                    f"n_original={len(by_cls[cls])})"
                )

        expected = sum(
            max(
                min(self.min_per_class, len(by_cls[c])),
                int(len(by_cls[c]) * self.fraction),
            )
            for c in by_cls
        )
        tolerance = max(5, int(0.05 * expected))
        total_sampled = len(sampled)
        if abs(total_sampled - expected) > tolerance:
            raise ValueError(
                f"StratifiedDeviceSampler produced {total_sampled} samples "
                f"but expected ~{expected} (tolerance {tolerance}); "
                f"fraction={self.fraction}, min_per_class={self.min_per_class}"
            )

    def summary(self) -> dict[str, Any]:
        """
        Return a summary of the most recent ``sample()`` call.

        Returns
        -------
        dict[str, Any]
            Keys: ``total_original``, ``total_sampled``, ``fraction_achieved``,
            ``min_per_class``, ``max_per_class``, ``classes_total``,
            ``classes_below_floor``, plus ``per_class_kept`` and
            ``per_class_original`` breakdowns for logging.

        Raises
        ------
        RuntimeError
            If called before ``sample()``.
        """
        if self._summary is None:
            raise RuntimeError(
                "StratifiedDeviceSampler.summary() called before sample()"
            )
        return dict(self._summary)
