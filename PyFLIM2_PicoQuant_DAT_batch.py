# -*- coding: utf-8 -*-
"""
PyFLIM2, PicoQuant .dat histogram batch loader

This script supports two common input layouts:

1) Legacy Tri2-style: folders named by condition (FP_order items) containing per-FOV
   histogram files with two columns: time<TAB>count (and one header row to skip).

2) PicoQuant "All Histograms" / "*Histo*.dat" exports: a single TSV-like file containing
   multiple histograms (typically FOV_1, FOV_2, ...), with the time column immediately
   to the left of each FOV column.

It produces:
- df: long-form per-FOV histogram data
- summary: per-condition summed histogram (normalised)
- peak_summary: per-FOV mean lifetime and FRET efficiency (relative to donor)
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Calcs:
    def __init__(self, t_D: float, t_DA: float):
        self.t_D = float(t_D)
        self.t_DA = float(t_DA)

    def fret_eff(self) -> float:
        return 1.0 - (self.t_DA / self.t_D)

    def dist_D_DA(self, r0: float) -> float:
        # R = r0 * ((1-E)/E)^(1/6)
        E = self.fret_eff()
        return (((1.0 - E) / E) ** (1.0 / 6.0)) * float(r0)

    def et_rate(self) -> float:
        # k_ET = 1/t_DA - 1/t_D
        return (1.0 / (self.t_DA * 1e-9)) - (1.0 / (self.t_D * 1e-9))


def _condition_from_filename(path: Path) -> str:
    name = path.stem
    name = re.sub(r"\s*[-_ ]*\s*Histo\s*$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def read_pq_histodat(path: str | Path) -> pd.DataFrame:
    """
    Read PicoQuant histogram export where each FOV column has a Lifetime column
    immediately to its left. Returns long-form DF with columns:
      condition, filename, time, count, source_file
    """
    path = Path(path)

    with open(path, "r", encoding="cp1252", errors="replace") as f:
        lines = f.read().splitlines()

    if len(lines) < 3:
        raise ValueError(f"{path.name} looks too short to be a PicoQuant histogram export.")

    # PicoQuant exports commonly use two header rows. The second header row carries the usable names.
    header2 = lines[1].split("\t")

    # Find (time_idx, count_idx) pairs by scanning for FOV columns
    pairs: list[tuple[int, int, str]] = []
    for j, token in enumerate(header2):
        tok = (token or "").strip()
        if tok.startswith("FOV_"):
            i = j - 1  # time column is usually immediately left of FOV
            if i >= 0:
                m = re.match(r"^(FOV_\d+)", tok)
                pairs.append((i, j, m.group(1) if m else tok))

    if not pairs:
        raise ValueError(
            f"Could not find any FOV_* columns in {path.name}. "
            "If this file is a fit-parameter table, it needs a different parser."
        )

    rows = [ln.split("\t") for ln in lines[2:] if ln.strip()]
    maxlen = max(len(r) for r in rows)
    rows = [r + [""] * (maxlen - len(r)) for r in rows]

    condition = _condition_from_filename(path)

    parts = []
    for time_idx, count_idx, fov in pairs:
        t = pd.to_numeric([r[time_idx] for r in rows], errors="coerce")
        c = pd.to_numeric([r[count_idx] for r in rows], errors="coerce")

        part = pd.DataFrame(
            {
                "Fluorophore": condition,
                "filename": fov,
                "time": t,
                "count": c,
                "source_file": path.name,
            }
        ).dropna(subset=["time", "count"])

        parts.append(part)

    out = pd.concat(parts, ignore_index=True)
    out = out[out["time"] >= 0].reset_index(drop=True)
    return out


def read_legacy_folder(folder: str | Path, condition: str) -> pd.DataFrame:
    """
    Read legacy Tri2-style folder containing per-FOV histogram files.
    Each file: one header row, then time<TAB>count.
    """
    folder = Path(folder)
    parts = []
    for fn in sorted(folder.iterdir()):
        if not fn.is_file():
            continue
        # Read 2-column table
        tmp = pd.read_csv(fn, sep="\t", skiprows=1, names=["time", "count"])
        tmp["filename"] = fn.name
        tmp["Fluorophore"] = condition
        tmp["source_file"] = fn.name
        parts.append(tmp[["Fluorophore", "filename", "time", "count", "source_file"]])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["Fluorophore", "filename", "time", "count", "source_file"]
    )


def load_all_data(
    *,
    fp_order: list[str] | None = None,
    dat_folder: str | Path = ".",
    dat_pattern: str = "*Histo*.dat",
) -> pd.DataFrame:
    """
    If fp_order folders exist, load legacy data from them.
    Also loads PicoQuant histogram .dat files from dat_folder matching dat_pattern.
    Returns concatenated long-form dataframe.
    """
    all_parts = []

    # 1) Legacy folders
    if fp_order:
        for cond in fp_order:
            if Path(cond).is_dir():
                df_leg = read_legacy_folder(cond, cond)
                if len(df_leg):
                    all_parts.append(df_leg)

    # 2) PicoQuant histogram .dat files
    dat_folder = Path(dat_folder)
    for p in sorted(dat_folder.glob(dat_pattern)):
        if p.is_file():
            all_parts.append(read_pq_histodat(p))

    if not all_parts:
        raise ValueError(
            "No input data found. Either:\n"
            "- create condition folders listed in FP_order containing histogram files, or\n"
            "- place PicoQuant '*Histo*.dat' files in the working directory (or set dat_folder/dat_pattern)."
        )

    return pd.concat(all_parts, ignore_index=True)


def main() -> None:
    # If you are working purely with PicoQuant .dat files, FP_order can be left None.
    # If you want a fixed order for plotting, set FP_order explicitly.
    FP_order = None  # e.g. ['GFP-vinc', 'GFP-vinc + mScar-RIAM', 'GFP-vinc + mScar-RIAM + Noc']

    # Load everything in the current directory that looks like a PicoQuant histogram export
    df = load_all_data(fp_order=FP_order, dat_folder=".", dat_pattern="*Histo*.dat")

    # If you want to enforce a custom order, set FP_order above and uncomment:
    # df["Fluorophore"] = pd.Categorical(df["Fluorophore"], categories=FP_order, ordered=True)

    # Normalisations
    df["norm_cc"] = df.groupby(["Fluorophore", "filename"])["count"].transform(
        lambda x: x / x.max() if float(x.max()) != 0.0 else x
    )
    df["per_sample"] = df.groupby("Fluorophore")["count"].transform(
        lambda x: x / x.max() if float(x.max()) != 0.0 else x
    )

    # Summed histogram per condition
    summary = (
        df.groupby(["Fluorophore", "time"], as_index=False)["norm_cc"]
          .sum()
          .rename(columns={"norm_cc": "sum"})
    )
    summary["norm_sum_per_fp"] = summary.groupby("Fluorophore")["sum"].transform(
        lambda x: x / x.max() if float(x.max()) != 0.0 else x
    )

    # Mean lifetime per FOV, intensity-weighted using raw counts
    peak_summary = (
        df.groupby(["Fluorophore", "filename"], as_index=False)
          .apply(lambda g: pd.Series({
              "mean lifetime": (g["time"] * g["count"]).sum() / g["count"].sum() if float(g["count"].sum()) != 0.0 else float("nan")
          }))
          .reset_index(drop=True)
    )

    # Donor reference for FRET efficiency, choose the condition with the largest mean lifetime
    ave = peak_summary.groupby("Fluorophore")["mean lifetime"].mean().sort_values(ascending=False)
    donor = str(ave.index[0])
    tau_donor = float(ave.iloc[0])

    peak_summary["FRET_E"] = (1.0 - (peak_summary["mean lifetime"] / tau_donor)) * 100.0

    # Save outputs
    df.to_csv("all_histograms_longform.csv", index=False)
    summary.to_csv("summary.csv", index=False)
    peak_summary.to_csv("peak_summary.csv", index=False)

    print(f"Loaded {df['source_file'].nunique()} source files")
    print(f"Found {df['Fluorophore'].nunique()} conditions, {df['filename'].nunique()} FOV traces")
    print(f"Using donor reference: {donor} (mean lifetime {tau_donor:.3f} ns)")

    # Simple plots (optional)
    sns.set_context("talk")

    plt.figure()
    sns.lineplot(data=summary, x="time", y="norm_sum_per_fp", hue="Fluorophore")
    plt.xlabel("Time (ns)")
    plt.ylabel("Normalised summed pixel frequency")
    plt.title("Average decay per condition")
    plt.tight_layout()

    plt.figure()
    sns.boxplot(data=peak_summary, x="Fluorophore", y="mean lifetime")
    plt.xticks(rotation=30, ha="right")
    plt.title("Mean lifetime per FOV")
    plt.tight_layout()

    plt.figure()
    sns.boxplot(data=peak_summary, x="Fluorophore", y="FRET_E")
    plt.xticks(rotation=30, ha="right")
    plt.title("FRET efficiency per FOV (relative to donor)")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
