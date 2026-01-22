# -*- coding: utf-8 -*-
"""
PyFLIM2 PicoQuant .dat histogram batch loader, with a simple GUI (tkinter).

Loads multiple PicoQuant histogram exports (for example "*Histo*.dat") from a folder.
Each .dat file becomes one condition (derived from filename), and each FOV_* column becomes one trace.

Outputs:
- all_histograms_longform.csv
- summary.csv
- peak_summary.csv

tkinter ships with standard Python on Windows.
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception:
    sns = None

import tkinter as tk
from tkinter import ttk, filedialog, messagebox


def _condition_from_filename(path: Path) -> str:
    name = path.stem
    name = re.sub(r"\s*[-_ ]*\s*Histo\s*$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def read_pq_histodat(path: str | Path) -> pd.DataFrame:
    """
    Read PicoQuant histogram export where each FOV column has a Lifetime column immediately to its left.
    Returns long-form DF with columns:
      Fluorophore, filename, time, count, source_file
    """
    path = Path(path)

    with open(path, "r", encoding="cp1252", errors="replace") as f:
        lines = f.read().splitlines()

    if len(lines) < 3:
        raise ValueError(f"{path.name} looks too short to be a PicoQuant histogram export.")

    header2 = lines[1].split("\t")

    pairs: list[tuple[int, int, str]] = []
    for j, token in enumerate(header2):
        tok = (token or "").strip()
        if tok.startswith("FOV_"):
            i = j - 1
            if i >= 0:
                m = re.match(r"^(FOV_\d+)", tok)
                pairs.append((i, j, m.group(1) if m else tok))

    if not pairs:
        raise ValueError(
            f"Could not find any FOV_* columns in {path.name}. "
            "This file may be a fit-parameter table rather than a histogram export."
        )

    rows = [ln.split("\t") for ln in lines[2:] if ln.strip()]
    if not rows:
        raise ValueError(f"{path.name} contained no data rows.")

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


def load_many_pq_histodats(folder: str | Path, pattern: str) -> pd.DataFrame:
    folder = Path(folder)
    paths = sorted(folder.glob(pattern))
    if not paths:
        raise ValueError(f"No files matched pattern '{pattern}' in: {folder}")

    dfs = []
    for p in paths:
        if p.is_file():
            dfs.append(read_pq_histodat(p))
    return pd.concat(dfs, ignore_index=True)


def compute_outputs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    df["norm_cc"] = df.groupby(["Fluorophore", "filename"])["count"].transform(
        lambda x: x / x.max() if float(x.max()) != 0.0 else x
    )
    df["per_sample"] = df.groupby("Fluorophore")["count"].transform(
        lambda x: x / x.max() if float(x.max()) != 0.0 else x
    )

    summary = (
        df.groupby(["Fluorophore", "time"], as_index=False)["norm_cc"]
          .sum()
          .rename(columns={"norm_cc": "sum"})
    )
    summary["norm_sum_per_fp"] = summary.groupby("Fluorophore")["sum"].transform(
        lambda x: x / x.max() if float(x.max()) != 0.0 else x
    )

    peak_summary = (
        df.groupby(["Fluorophore", "filename"], as_index=False)
          .apply(lambda g: pd.Series({
              "mean lifetime": (g["time"] * g["count"]).sum() / g["count"].sum()
              if float(g["count"].sum()) != 0.0 else float("nan")
          }))
          .reset_index(drop=True)
    )

    ave = peak_summary.groupby("Fluorophore")["mean lifetime"].mean().sort_values(ascending=False)
    tau_donor = float(ave.iloc[0])
    peak_summary["FRET_E"] = (1.0 - (peak_summary["mean lifetime"] / tau_donor)) * 100.0

    return summary, peak_summary


@dataclass
class RunConfig:
    folder: Path
    pattern: str
    out_dir: Path
    make_plots: bool


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("PyFLIM2 PicoQuant .dat Loader")
        self.geometry("860x560")

        self.folder_var = tk.StringVar(value=str(Path.cwd()))
        self.pattern_var = tk.StringVar(value="*Histo*.dat")
        self.outdir_var = tk.StringVar(value=str(Path.cwd()))
        self.plots_var = tk.BooleanVar(value=True)

        self._build()
        self._set_running(False)

    def _build(self) -> None:
        pad = {"padx": 10, "pady": 6}

        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True)

        inputs = ttk.LabelFrame(frm, text="Inputs")
        inputs.pack(fill="x", **pad)

        ttk.Label(inputs, text="Folder with .dat files").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(inputs, textvariable=self.folder_var, width=70).grid(row=0, column=1, sticky="we", **pad)
        ttk.Button(inputs, text="Browse", command=self._browse_folder).grid(row=0, column=2, sticky="e", **pad)

        ttk.Label(inputs, text="Filename pattern").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(inputs, textvariable=self.pattern_var, width=25).grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(inputs, text="Output folder").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(inputs, textvariable=self.outdir_var, width=70).grid(row=2, column=1, sticky="we", **pad)
        ttk.Button(inputs, text="Browse", command=self._browse_outdir).grid(row=2, column=2, sticky="e", **pad)

        ttk.Checkbutton(inputs, text="Make quick plots", variable=self.plots_var).grid(row=3, column=1, sticky="w", **pad)

        inputs.columnconfigure(1, weight=1)

        actions = ttk.Frame(frm)
        actions.pack(fill="x", **pad)

        self.run_btn = ttk.Button(actions, text="Run", command=self._run_clicked)
        self.run_btn.pack(side="left", padx=6)

        self.open_out_btn = ttk.Button(actions, text="Open output folder", command=self._open_outdir)
        self.open_out_btn.pack(side="left", padx=6)

        self.quit_btn = ttk.Button(actions, text="Quit", command=self.destroy)
        self.quit_btn.pack(side="right", padx=6)

        log_frame = ttk.LabelFrame(frm, text="Log")
        log_frame.pack(fill="both", expand=True, **pad)

        self.log = tk.Text(log_frame, height=18, wrap="word")
        self.log.pack(fill="both", expand=True, padx=10, pady=8)

        self.prog = ttk.Progressbar(frm, mode="indeterminate")
        self.prog.pack(fill="x", padx=12, pady=4)

    def _browse_folder(self) -> None:
        p = filedialog.askdirectory(initialdir=self.folder_var.get() or str(Path.cwd()))
        if p:
            self.folder_var.set(p)

    def _browse_outdir(self) -> None:
        p = filedialog.askdirectory(initialdir=self.outdir_var.get() or str(Path.cwd()))
        if p:
            self.outdir_var.set(p)

    def _open_outdir(self) -> None:
        out_dir = Path(self.outdir_var.get())
        if not out_dir.exists():
            messagebox.showerror("Output folder missing", "The selected output folder does not exist.")
            return
        try:
            os.startfile(out_dir)  # Windows
        except Exception:
            try:
                import subprocess
                subprocess.run(["xdg-open", str(out_dir)], check=False)
            except Exception:
                messagebox.showinfo("Open folder", f"Output folder: {out_dir}")

    def _append_log(self, msg: str) -> None:
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.update_idletasks()

    def _set_running(self, running: bool) -> None:
        self.run_btn.configure(state="disabled" if running else "normal")
        self.open_out_btn.configure(state="disabled" if running else "normal")
        if running:
            self.prog.start(10)
        else:
            self.prog.stop()

    def _run_clicked(self) -> None:
        cfg = RunConfig(
            folder=Path(self.folder_var.get()),
            pattern=self.pattern_var.get().strip() or "*Histo*.dat",
            out_dir=Path(self.outdir_var.get()),
            make_plots=bool(self.plots_var.get()),
        )

        if not cfg.folder.exists():
            messagebox.showerror("Folder missing", "Input folder does not exist.")
            return
        if not cfg.out_dir.exists():
            messagebox.showerror("Folder missing", "Output folder does not exist.")
            return

        self._set_running(True)
        self._append_log("Starting...")

        th = threading.Thread(target=self._worker, args=(cfg,), daemon=True)
        th.start()

    def _worker(self, cfg: RunConfig) -> None:
        try:
            self._append_log(f"Loading files from: {cfg.folder}")
            self._append_log(f"Pattern: {cfg.pattern}")

            df = load_many_pq_histodats(cfg.folder, cfg.pattern)

            self._append_log(f"Loaded {df['source_file'].nunique()} source files")
            self._append_log(f"Conditions: {df['Fluorophore'].nunique()}, FOV traces: {df['filename'].nunique()}")

            summary, peak_summary = compute_outputs(df)

            out_long = cfg.out_dir / "all_histograms_longform.csv"
            out_sum = cfg.out_dir / "summary.csv"
            out_peak = cfg.out_dir / "peak_summary.csv"

            df.to_csv(out_long, index=False)
            summary.to_csv(out_sum, index=False)
            peak_summary.to_csv(out_peak, index=False)

            self._append_log("Saved:")
            self._append_log(f"  {out_long}")
            self._append_log(f"  {out_sum}")
            self._append_log(f"  {out_peak}")

            if cfg.make_plots:
                self._append_log("Making plots...")
                self._make_plots(summary, peak_summary)

            self._append_log("Done.")
            self.after(0, lambda: self._set_running(False))

        except Exception as e:
            self.after(0, lambda: self._set_running(False))
            self.after(0, lambda: messagebox.showerror("Run failed", str(e)))
            self._append_log(f"ERROR: {e}")

    def _make_plots(self, summary: pd.DataFrame, peak_summary: pd.DataFrame) -> None:
        if sns is not None:
            sns.set_context("talk")

        plt.figure()
        if sns is not None:
            sns.lineplot(data=summary, x="time", y="norm_sum_per_fp", hue="Fluorophore")
        else:
            for cond, g in summary.groupby("Fluorophore"):
                plt.plot(g["time"], g["norm_sum_per_fp"], label=str(cond))
            plt.legend()
        plt.xlabel("Time (ns)")
        plt.ylabel("Normalised summed pixel frequency")
        plt.title("Average decay per condition")
        plt.tight_layout()

        plt.figure()
        if sns is not None:
            sns.boxplot(data=peak_summary, x="Fluorophore", y="mean lifetime")
            plt.xticks(rotation=30, ha="right")
        else:
            order = list(peak_summary["Fluorophore"].unique())
            data = [peak_summary.loc[peak_summary["Fluorophore"] == o, "mean lifetime"].values for o in order]
            plt.boxplot(data, labels=order)
            plt.xticks(rotation=30, ha="right")
        plt.title("Mean lifetime per FOV")
        plt.tight_layout()

        plt.figure()
        if sns is not None:
            sns.boxplot(data=peak_summary, x="Fluorophore", y="FRET_E")
            plt.xticks(rotation=30, ha="right")
        else:
            order = list(peak_summary["Fluorophore"].unique())
            data = [peak_summary.loc[peak_summary["Fluorophore"] == o, "FRET_E"].values for o in order]
            plt.boxplot(data, labels=order)
            plt.xticks(rotation=30, ha="right")
        plt.title("FRET efficiency per FOV (relative to donor)")
        plt.tight_layout()

        plt.show()


if __name__ == "__main__":
    app = App()
    app.mainloop()
