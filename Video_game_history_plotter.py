#!/usr/bin/env python3
import os, sys, re, glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ---------- Config ----------
COVERS_DIR = "Game_covers"
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".jfif"]

# Extreme wide canvas (great for zoom + horizontal scroll)
FIG_WIDTH_IN = 100   # try 60–120 if you want smaller/bigger
FIG_HEIGHT_IN = 6
EXPORT_DPI = 200     # total pixels ~ width_in * dpi
XTICK_FONTSIZE = 8
IMAGE_ZOOM = 0.28

OUTPUT_PNG = "ranking_plot_wide.png"
OUTPUT_SVG = "ranking_plot_wide.svg"

DEBUG_LIMIT = 0  # set >0 to print early debug

# ---------- CSV ----------
def safe_read_csv(path):
    return pd.read_csv(path, sep=None, engine="python", encoding="utf-8", skipinitialspace=True)

def build_game_to_image_map(df):
    game_to_path = {}
    if "Game" in df.columns and "Image_path" in df.columns:
        for _, row in df[["Game", "Image_path"]].dropna().iterrows():
            g = str(row["Game"]).strip()
            p = str(row["Image_path"]).strip()
            if g:
                game_to_path[g] = p
    return game_to_path

# ---------- Filename helpers ----------
UNSAFE = r"""[:/\\?*"<>|’'“”‘—–\-.,!()&+]"""

def underscores(s: str):
    s = s.replace(" ", "_")
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def strip_punct(s: str):
    return re.sub(f"[{re.escape(UNSAFE)}]", "", s)

def slugify(s: str):
    s = underscores(s)
    s = strip_punct(s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def candidate_cores(title_or_path: str):
    raw = title_or_path.strip()
    u = underscores(raw)
    sl = slugify(raw)
    variants = [raw, u, sl, raw.lower(), u.lower(), sl.lower()]
    cores = []
    for v in variants:
        cores.append(v)
        cores.append(os.path.join(COVERS_DIR, v))
    seen, out = set(), []
    for c in cores:
        if c not in seen:
            seen.add(c)
            out.append(c)
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
    for ext in IMAGE_EXTENSIONS:
        cands.extend(glob.glob(pattern + ext))
    return cands[0] if cands else None

def read_image(path: str) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGBA"))

def label_for_period(year, q): return f"{int(year)} Q{int(q)}"

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # CSV
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csvs = [f for f in os.listdir(script_dir) if f.lower().endswith(".csv")]
        if not csvs:
            print("No CSV file found next to the script.")
            sys.exit(1)
        csv_path = os.path.join(script_dir, csvs[0])

    df = safe_read_csv(csv_path)
    required = {"Year", "Q", "Rank 1", "Rank 2", "Rank 3", "Rank 4"}
    missing = required - set(df.columns)
    if missing:
        print(f"Missing required columns: {missing}")
        sys.exit(1)

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["Q"] = pd.to_numeric(df["Q"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["Year", "Q"]).copy().sort_values(["Year", "Q"])

    df["PeriodLabel"] = df.apply(lambda r: label_for_period(r["Year"], r["Q"]), axis=1)
    periods = df["PeriodLabel"].unique().tolist()
    x_index = {p: i for i, p in enumerate(periods)}

    game_to_img = build_game_to_image_map(df)

    # --- super-wide figure ---
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    ax.set_title("Quarterly Rankings (1 = Top, 4 = Bottom)")
    ax.set_xlabel("Time (Year • Quarter)")
    ax.set_ylabel("Rank")
    ax.set_ylim(4.5, 0.5)
    ax.set_yticks([1, 2, 3, 4])
    ax.set_xlim(-0.5, len(periods) - 0.5)
    ax.set_xticks(range(len(periods)))
    ax.set_xticklabels(periods, rotation=45, ha="right", fontsize=XTICK_FONTSIZE)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    rank_cols = ["Rank 1", "Rank 2", "Rank 3", "Rank 4"]
    missing_titles, debug_count = set(), 0

    for _, row in df.iterrows():
        x = x_index[row["PeriodLabel"]]
        for rc in rank_cols:
            game = str(row[rc]).strip()
            if not game or game == "nan": continue
            try: rank_num = int(rc.split()[-1])
            except: rank_num = rank_cols.index(rc) + 1

            base = game_to_img.get(game, game)
            path = find_best_image(base, script_dir)

            if path:
                try:
                    img = read_image(path)
                    ab = AnnotationBbox(OffsetImage(img, zoom=IMAGE_ZOOM), (x, rank_num), frameon=False)
                    ax.add_artist(ab)
                    if debug_count < DEBUG_LIMIT:
                        print(f"[OK] {row['PeriodLabel']} r{rank_num} '{game}' -> {os.path.relpath(path, script_dir)}")
                        debug_count += 1
                except Exception as e:
                    if debug_count < DEBUG_LIMIT:
                        print(f"[ERR] Could not read image for '{game}' at {path}: {e}")
                        debug_count += 1
                    ax.text(x, rank_num, game, fontsize=8, ha="center", va="center")
                    missing_titles.add(game)
            else:
                if debug_count < DEBUG_LIMIT:
                    print(f"[MISS] {row['PeriodLabel']} r{rank_num} '{game}' -> no file matched")
                    debug_count += 1
                ax.text(x, rank_num, game, fontsize=8, ha="center", va="center")
                missing_titles.add(game)

    # Don’t let tight_layout shrink width; stick to our big canvas
    fig.subplots_adjust(left=0.06, right=0.995, top=0.90, bottom=0.22)

    # Save huge PNG and crisp SVG
    out_png = os.path.join(script_dir, OUTPUT_PNG)
    out_svg = os.path.join(script_dir, OUTPUT_SVG)
    fig.savefig(out_png, dpi=EXPORT_DPI)
    fig.savefig(out_svg)  # vector; great for zooming
    print(f"Saved wide exports:\n - {out_png}\n - {out_svg}")

    if missing_titles:
        print("\nCovers not found for these titles:")
        for t in sorted(missing_titles):
            print(" -", t)

    # Optional: comment out if you don’t want an interactive window
    # plt.show()

if __name__ == "__main__":
    main()
