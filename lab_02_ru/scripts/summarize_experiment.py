#!/usr/bin/env python3
"""Дополнительные таблицы и графики для ответов на вопросы Lab 02."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from plot_augmentation_features import log_mel

LAB_DIR = Path(__file__).resolve().parents[1]


def write_durations(out_dir: Path) -> None:
    rows = []
    for path in sorted((LAB_DIR / "data/wav").glob("*.wav")):
        row: dict[str, str | float] = {"item_id": path.stem}
        for condition in ("clean", "speed_0.9x", "speed_1.1x"):
            info = sf.info(out_dir / "augmented" / f"{path.stem}__{condition}.wav")
            row[condition] = round(info.frames / info.samplerate, 6)
        rows.append(row)
    results = out_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    with (results / "duration_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_conditions(out_dir: Path, item_id: str) -> None:
    conditions = ("clean", "noise_10db", "reverb_room_a")
    fig, axes = plt.subplots(len(conditions), 2, figsize=(13, 9))
    for row, condition in enumerate(conditions):
        audio, sr = sf.read(out_dir / "augmented" / f"{item_id}__{condition}.wav", dtype="float32")
        librosa.display.waveshow(audio, sr=sr, ax=axes[row, 0])
        axes[row, 0].set(xlim=(0, len(audio) / sr), ylim=(-1, 1), title=f"{condition}: waveform", xlabel="Time (s)", ylabel="Amplitude")
        image = librosa.display.specshow(log_mel(audio, sr), sr=sr, hop_length=160,
            x_axis="time", y_axis="mel", fmin=20, fmax=7600,
            ax=axes[row, 1], cmap="viridis", vmin=-80, vmax=0)
        axes[row, 1].set(xlim=(0, len(audio) / sr), title=f"{condition}: log-Mel", xlabel="Time (s)", ylabel="Hz (Mel scale)")
        fig.colorbar(image, ax=axes[row, 1], format="%+2.0f dB")
    fig.suptitle(f"Noise and reverberation: {item_id}")
    fig.tight_layout()
    fig.savefig(out_dir / "figures/noise_vs_reverb.png", dpi=160)
    plt.close(fig)


def plot_metrics(out_dir: Path) -> None:
    source = out_dir / "results/aggregate_results.csv"
    if not source.exists():
        return
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: float(row["mean_wer"]))
    indices = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, key, label, color in ((-.18, "mean_wer", "Mean WER", "#2463a6"), (.18, "mean_cer", "Mean CER", "#d18427")):
        bars = ax.barh(indices + offset, [100 * float(row[key]) for row in rows], height=.34, label=label, color=color)
        ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.set_yticks(indices, [row["condition"] for row in rows])
    ax.invert_yaxis()
    ax.set(xlabel="Error rate (%)", title="tiny.en / CPU int8: mean of five utterances")
    ax.set_xlim(0, max(100 * float(row[key]) for row in rows for key in ("mean_wer", "mean_cer")) * 1.2)
    ax.legend(loc="upper right")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(out_dir / "figures/aggregate_error_rates.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=LAB_DIR / "outputs")
    parser.add_argument("--item-id", default="1272-128104-0000")
    args = parser.parse_args()
    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)
    write_durations(args.out_dir)
    plot_conditions(args.out_dir, args.item_id)
    plot_metrics(args.out_dir)
    print(f"Saved duration table and comparison figures to {args.out_dir}")


if __name__ == "__main__":
    main()
