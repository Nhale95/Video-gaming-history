#!/usr/bin/env python3
import os, sys, re, glob, hashlib
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import numpy as np
from PIL import Image

# ---------- Global dark theme ----------
plt.style.use("dark_background")

mpl.rcParams.update({
    "axes.facecolor": "#1e1e1e",
    "figure.facecolor": "#1e1e1e",
    "axes.edgecolor": "#aaaaaa",
    "axes.labelcolor": "#e0e0e0",
    "text.color": "#e0e0e0",
    "xtick.color": "#cccccc",
    "ytick.color": "#cccccc",
    "grid.color": "#444444",
    "grid.linestyle": "--",
    "axes.grid": True,
    "legend.facecolor": "#2b2b2b",
    "legend.edgecolor": "none",
    "legend.labelcolor": "#e0e0e0",
    "font.family": "DejaVu Sans",
})

# ---------- Matplotlib tweaks ----------
mpl.rcParams["hatch.linewidth"] = 1.5  # bolder hatch lines for clarity

# ---------- Config ----------
COVERS_DIR = "Game_covers"
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".jfif"]

FIG_WIDTH_IN = 100
FIG_HEIGHT_IN_MAIN = 5.0    # middle: covers axis height
FIG_HEIGHT_IN_MINI = 1.6    # bottom: minutes axis height
EXPORT_DPI = 200
XTICK_FONTSIZE = 8
OUTPUT_PNG = "ranking_plot_wide.png"

# Fixed data-space “slot” per cover
SLOT_W = 0.9
SLOT_H = 0.8
BORDER_PAD_FRAC = 0.015

# ---------- Distinct console colors ----------
CONSOLE_COLORS = {
    "Nintendo 64":        "#1b9e77",
    "PC":                 "#000000",
    "PS1":                "#386cb0",
    "Game Boy Color":     "#7b3294",
    "PS2":                "#4daf4a",
    "Nintendo Game Boy":  "#e41a1c",
    "Game Boy Advance":   "#ff7f00",
    "Game Cube":          "#a65628",
    "Xbox":               "#66a61e",
    "Xbox 360":           "#006d2c",
    "Wii":                "#80cdc1",
    "PS4":                "#253494",
    "Nintendo Switch":    "#b15928",
    "Switch":             "#b15928",
    "PlayStation":        "#386cb0",
    "PlayStation 2":      "#4daf4a",
}
_fallback_palette = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

HATCH_PATTERNS = [
    "///", "\\\\\\", "xxx", "++", "--", "||", "..", "oo", "**", "//-",
    "\\\\|", "/x", "\\+", "|o", ".-", "o*", "x-", "+|", "o.", "*|"
]

# Period colors
PERIOD_COLORS = {
    "PreSchool":          "#b7dbf2",
    "Primary School":     "#d2c3ff",
    "Secondary School":   "#ffcf8a",
    "University":         "#bfe5b1",
    "Early Career":       "#ffb7c6",
    "Adulthood":          "#bcdcff",
}
_period_fallback = ["#d9d9d9", "#c7c7c7"]

# ---------- Helpers ----------
def safe_read_csv(path):
    return pd.read_csv(path, sep=None, engine="python", encoding="utf-8", skipinitialspace=True)

def _norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip()).lower()

def build_game_maps(df: pd.DataFrame):
    game_to_img, game_to_console = {}, {}
    def add_row(title, console=None, image_core=None):
        if not isinstance(title, str) or not title.strip(): return
        k = _norm_key(title)
        if isinstance(console, str) and console.strip():
            game_to_console[k] = console.strip()
        if isinstance(image_core, str) and image_core.strip():
            game_to_img[k] = image_core.strip()

    has_game = "Game" in df.columns
    has_console = "Console" in df.columns
    has_img = "Image_path" in df.columns
    if has_game and (has_console or has_img):
        cols = ["Game"] + (["Console"] if has_console else []) + (["Image_path"] if has_img else [])
        for _, r in df[cols].dropna(how="all").iterrows():
            add_row(r.get("Game", ""), r.get("Console", None), r.get("Image_path", None))

    if len(df.columns) >= 3:
        a, b, c = df.columns[-3], df.columns[-2], df.columns[-1]
        tail = df[[a, b, c]].copy().dropna(how="all")
        for _, r in tail.iterrows():
            add_row(r.get(a, ""), r.get(b, None), r.get(c, None))

    return game_to_img, game_to_console

UNSAFE = r"""[:/\\?*"<>|’'“”‘—–\-.,!()&+]"""
def underscores(s: str): return re.sub(r"_+", "_", s.replace(" ", "_")).strip("_")
def strip_punct(s: str): return re.sub(f"[{re.escape(UNSAFE)}]", "", s)
def slugify(s: str): return re.sub(r"_+", "_", strip_punct(underscores(s))).strip("_")

def candidate_cores(title_or_path: str):
    raw = title_or_path.strip()
    variants = [raw, underscores(raw), slugify(raw),
                raw.lower(), underscores(raw).lower(), slugify(raw).lower()]
    cores = []
    for v in variants:
        cores.append(v); cores.append(os.path.join(COVERS_DIR, v))
    seen, out = set(), []
    for c in cores:
        if c not in seen:
            seen.add(c); out.append(c)
    return out

def find_best_image(base_core: str, script_dir: str):
    for core in candidate_cores(base_core):
        if core.lower().endswith(tuple(IMAGE_EXTENSIONS)):
            p = os.path.join(script_dir, core)
            if os.path.isfile(p): return p
        else:
            for ext in IMAGE_EXTENSIONS:
                p = os.path.join(script_dir, core + ext)
                if os.path.isfile(p): return p
    slug = slugify(base_core)
    pattern = os.path.join(script_dir, COVERS_DIR, f"{slug}*")
    cands = []
    for ext in IMAGE_EXTENSIONS: cands.extend(glob.glob(pattern + ext))
    return cands[0] if cands else None

def read_image_rgba(path: str) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGBA"))

def display_console_name(console: str) -> str:
    if not console: return "Unknown"
    c = str(console).strip()
    if re.match(r"(?i)^windows\b", c): return "PC"
    return c

def color_for_console(console: str) -> str:
    if not console: return "gray"
    c = display_console_name(console)
    return CONSOLE_COLORS.get(c, "#888888")

def label_for_period(year, q): return f"{int(year)} Q{int(q)}"

def compute_period_spans(df_sorted, x_index):
    periods = df_sorted["Period"].astype(str).tolist()
    labels = df_sorted["PeriodLabel"].tolist()
    spans = []
    start_i = 0
    for i in range(1, len(periods)+1):
        if i == len(periods) or periods[i] != periods[start_i]:
            x0 = x_index[labels[start_i]] - 0.5
            x1 = x_index[labels[i-1]] + 0.5
            spans.append((x0, x1, periods[start_i]))
            start_i = i
    return spans

def shade_periods(ax, spans):
    for k, (x0, x1, period_name) in enumerate(spans):
        color = PERIOD_COLORS.get(period_name, _period_fallback[k % len(_period_fallback)])
        ax.axvspan(x0, x1, color=color, alpha=0.80, zorder=0, linewidth=0)

def label_periods_center(ax, spans):
    ymin, ymax = ax.get_ylim()
    ymid = (ymin + ymax) / 2.0
    for k, (x0, x1, period_name) in enumerate(spans):
        ax.text((x0+x1)/2, ymid, period_name,
                ha="center", va="center", fontsize=10, color="black",
                bbox=dict(facecolor="white", alpha=0.35, edgecolor="none", boxstyle="round,pad=0.2"),
                zorder=1)

def draw_cover(ax, img_rgba, center_x, center_y, slot_w, slot_h, edge_color, lw=2.0):
    H, W = img_rgba.shape[0], img_rgba.shape[1]
    img_aspect = W / H
    slot_aspect = slot_w / slot_h
    if img_aspect >= slot_aspect:
        draw_w = slot_w
        draw_h = draw_w / img_aspect
    else:
        draw_h = slot_h
        draw_w = draw_h * img_aspect
    x0 = center_x - draw_w/2
    y0 = center_y - draw_h/2
    ax.imshow(img_rgba, extent=(x0, x0+draw_w, y0, y0+draw_h),
              origin="lower", interpolation="antialiased", zorder=3)
    pad_w = draw_w * BORDER_PAD_FRAC
    pad_h = draw_h * BORDER_PAD_FRAC
    rect = Rectangle((x0 - pad_w, y0 - pad_h),
                     draw_w + 2*pad_w, draw_h + 2*pad_h,
                     fill=False, linewidth=lw, edgecolor=edge_color,
                     joinstyle="round", zorder=4)
    ax.add_patch(rect)

# ---------- main ----------
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = sys.argv[1] if len(sys.argv) > 1 else next((f for f in os.listdir(script_dir) if f.endswith(".csv")), None)
    if not csv_path:
        print("No CSV file found."); sys.exit(1)
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(script_dir, csv_path)

    df_all = safe_read_csv(csv_path)
    required = {"Year","Q","Rank 1","Rank 2","Rank 3","Rank 4","Average minutes played per day","Period"}
    miss = required - set(df_all.columns)
    if miss:
        print(f"Missing columns: {miss}"); sys.exit(1)

    df = df_all.copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["Q"] = pd.to_numeric(df["Q"], errors="coerce").astype("Int64")
    df["Average minutes played per day"] = pd.to_numeric(df["Average minutes played per day"], errors="coerce")
    df = df.dropna(subset=["Year","Q"]).copy().sort_values(["Year","Q"])

    df["PeriodLabel"] = df.apply(lambda r: label_for_period(r["Year"], r["Q"]), axis=1)
    periods = df["PeriodLabel"].unique().tolist()
    x_index = {p: i for i, p in enumerate(periods)}

    game_to_img, game_to_console = build_game_maps(df_all)

    # --- Layout: just two rows (covers + minutes) ---
    total_h = FIG_HEIGHT_IN_MAIN + FIG_HEIGHT_IN_MINI
    fig = plt.figure(figsize=(FIG_WIDTH_IN, total_h))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[FIG_HEIGHT_IN_MAIN, FIG_HEIGHT_IN_MINI])

    ax_main = fig.add_subplot(gs[0, 0])
    ax_min  = fig.add_subplot(gs[1, 0], sharex=ax_main)

    # --- Covers panel ---
    ax_main.set_title("Quarterly Rankings (1 = Top, 4 = Bottom)")
    ax_main.set_ylabel("Rank")
    ax_main.set_ylim(4.5, 0.5)
    ax_main.set_yticks([1,2,3,4])
    ax_main.set_xlim(-0.5, len(periods)-0.5)
    ax_main.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax_main.set_axisbelow(True)
    ax_main.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    rank_cols = ["Rank 1","Rank 2","Rank 3","Rank 4"]
    missing = set()
    for _, row in df.iterrows():
        x = x_index[row["PeriodLabel"]]
        for rc in rank_cols:
            name = str(row[rc]).strip()
            if not name or name.lower() == "nan": continue
            rank_num = int(rc.split()[-1])
            key = _norm_key(name)
            base = game_to_img.get(key, name)
            path = find_best_image(base, script_dir)
            raw_console = (game_to_console.get(key, "") or "Unknown")
            edge = color_for_console(raw_console)
            if path:
                try:
                    img = read_image_rgba(path)
                    draw_cover(ax_main, img, x, rank_num, SLOT_W, SLOT_H, edge_color=edge)
                except Exception:
                    ax_main.text(x, rank_num, name, fontsize=8, ha="center", va="center")
            else:
                missing.add(name)
                ax_main.text(x, rank_num, name, fontsize=8, ha="center", va="center")

    # --- Minutes panel ---
    spans = compute_period_spans(df, x_index)
    shade_periods(ax_min, spans)
    x_vals = [x_index[p] for p in df["PeriodLabel"]]
    y_vals = df["Average minutes played per day"].to_numpy()
    ax_min.plot(
    x_vals, y_vals,
    marker="o", markersize=5,
    linewidth=2.2, color="#4db6ac",  # teal-blue line
    markerfacecolor="#80cbc4", markeredgecolor="#004d40",
    zorder=3,
)
    ax_min.set_ylabel("Avg min/day")
    ax_min.set_xlabel("Time (Year • Quarter)")
    ax_min.set_xticks(range(len(periods)))
    ax_min.set_xticklabels(periods, rotation=45, ha="right", fontsize=XTICK_FONTSIZE)
    ax_min.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax_min.set_xlim(-0.5, len(periods)-0.5)
    label_periods_center(ax_min, spans)

    # --- Period legend ---
    periods_in_order = []
    for p in df["Period"].astype(str):
        if p not in periods_in_order:
            periods_in_order.append(p)
    period_handles = []
    for i, p in enumerate(periods_in_order):
        face = PERIOD_COLORS.get(p, _period_fallback[i % len(_period_fallback)])
        period_handles.append(Patch(facecolor=face, edgecolor="none", label=p, alpha=0.80))
    if period_handles:
        ax_min.legend(handles=period_handles, title="Periods",
                      loc="center left", bbox_to_anchor=(-0.026, 0.40),
                      frameon=False, handlelength=1.0, handletextpad=0.4,
                      labelspacing=0.3, borderpad=0.2, columnspacing=0.6)

    fig.subplots_adjust(left=0.031, right=0.995, top=0.94, bottom=0.20, hspace=0.05)
    for ax in [ax_main, ax_min]:
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.3)

        # --- Legends ---
    # Console legend (left of main axis)
    consoles_in_order = []
    for _, row in df.iterrows():
        for rc in ["Rank 1", "Rank 2", "Rank 3", "Rank 4"]:
            game = str(row[rc]).strip()
            if not game or game.lower() == "nan":
                continue
            key = _norm_key(game)
            raw_console = (game_to_console.get(key, "") or "Unknown")
            disp_console = display_console_name(raw_console)
            if disp_console not in consoles_in_order:
                consoles_in_order.append(disp_console)

    console_handles = []
    for c in consoles_in_order:
        col = color_for_console(c)
        console_handles.append(
            Patch(facecolor="white", edgecolor=col, linewidth=1.5, label=c)
        )

    if console_handles:
        ax_main.legend(
            handles=console_handles,
            title="Consoles",
            loc="center left",
            bbox_to_anchor=(-0.026, 0.55),
            frameon=False,
            handlelength=1.0,
            handletextpad=0.4,
            labelspacing=0.3,
            borderpad=0.2,
            columnspacing=0.6,
        )

    # Periods legend (left of bottom axis)
    periods_in_order = []
    for p in df["Period"].astype(str):
        if p not in periods_in_order:
            periods_in_order.append(p)
    period_handles = []
    for i, p in enumerate(periods_in_order):
        face = PERIOD_COLORS.get(p, _period_fallback[i % len(_period_fallback)])
        period_handles.append(Patch(facecolor=face, edgecolor="none", label=p, alpha=0.80))
    if period_handles:
        ax_min.legend(
            handles=period_handles,
            title="Periods",
            loc="center left",
            bbox_to_anchor=(-0.026, 0.40),
            frameon=False,
            handlelength=1.0,
            handletextpad=0.4,
            labelspacing=0.3,
            borderpad=0.2,
            columnspacing=0.6,
        )

    fig.text(0.01, 0.01, "Source: approximate gaming history", color="#888888", fontsize=8)
    out_png = os.path.join(script_dir, OUTPUT_PNG)
    fig.savefig(out_png, dpi=EXPORT_DPI)
    print(f"Saved: {out_png}")

    if missing:
        print("\nNo cover found for:")
        for t in sorted(missing): print(" -", t)

if __name__ == "__main__":
    main()
