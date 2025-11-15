#!/usr/bin/env python3
import os, sys, re
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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

# ---------- Config ----------
CONSOLE_COLORS = {
    "Nintendo 64": "#1b9e77",
    "PC": "#000000",
    "PS1": "#386cb0",
    "Game Boy Color": "#7b3294",
    "PS2": "#4daf4a",
    "Nintendo Game Boy": "#e41a1c",
    "Game Boy Advance": "#ff7f00",
    "Game Cube": "#a65628",
    "Xbox": "#66a61e",
    "Xbox 360": "#006d2c",
    "Wii": "#80cdc1",
    "PS4": "#253494",
    "Nintendo Switch": "#b15928",
    "Switch": "#b15928",
    "PlayStation": "#386cb0",
    "PlayStation 2": "#4daf4a",
}

CONSOLE_SHORT = {
    "Nintendo 64": "N64",
    "PC": "PC",
    "PS1": "PS1",
    "Game Boy Color": "GB Color",
    "PS2": "PS2",
    "Nintendo Game Boy": "GB",
    "Game Boy Advance": "GBA",
    "Game Cube": "GC",
    "Xbox": "Xbox",
    "Xbox 360": "X360",
    "Wii": "Wii",
    "PS4": "PS4",
    "Nintendo Switch": "Switch",
    "Switch": "Switch",
    "PlayStation": "PS1",
    "PlayStation 2": "PS2",
}

PERIOD_COLORS = {
    "PreSchool": "#b7dbf2",
    "Primary School": "#d2c3ff",
    "Secondary School": "#ffcf8a",
    "University": "#bfe5b1",
    "Early Career": "#ffb7c6",
    "Adulthood": "#bcdcff",
}
_period_fallback = ["#d9d9d9", "#c7c7c7"]

# ---------- Helpers ----------
def safe_read_csv(path):
    return pd.read_csv(path, sep=None, engine="python", encoding="utf-8", skipinitialspace=True)

def _norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip()).lower()

def display_console_name(console: str) -> str:
    if not console:
        return "Unknown"
    c = str(console).strip()
    if re.match(r"(?i)^windows\b", c):
        return "PC"
    return c

def short_console_name(console: str) -> str:
    c = display_console_name(console)
    return CONSOLE_SHORT.get(c, c)

def color_for_console(console: str) -> str:
    c = display_console_name(console)
    return CONSOLE_COLORS.get(c, "#888888")

# ---------- Main ----------
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = sys.argv[1] if len(sys.argv) > 1 else next((f for f in os.listdir(script_dir) if f.endswith(".csv")), None)
    if not csv_path:
        print("No CSV file found."); sys.exit(1)
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(script_dir, csv_path)

    df_all = safe_read_csv(csv_path)
    rank_cols = ["Rank 1", "Rank 2", "Rank 3", "Rank 4"]
    if not all(col in df_all.columns for col in rank_cols):
        print("Missing rank columns."); sys.exit(1)

    # Build game-to-console mapping
    game_to_console = {}
    if "Game" in df_all.columns and "Console" in df_all.columns:
        for _, r in df_all[["Game", "Console"]].dropna(how="any").iterrows():
            game_to_console[_norm_key(r["Game"])] = display_console_name(r["Console"])

    # Clean data
    df = df_all.copy()
    df["Average minutes played per day"] = pd.to_numeric(df["Average minutes played per day"], errors="coerce").fillna(0)
    df = df.dropna(subset=["Year", "Q"])

    df["Quarter_minutes"] = df["Average minutes played per day"] * 90
    rank_weights = {"Rank 1": 0.4, "Rank 2": 0.3, "Rank 3": 0.2, "Rank 4": 0.1}

    console_weights, period_weights, game_weights = {}, {}, {}
    for _, row in df.iterrows():
        quarter_minutes = row["Quarter_minutes"]
        period = str(row.get("Period", "Unknown")).strip()
        for rc in rank_cols:
            if rc not in row or pd.isna(row[rc]): continue
            game = str(row[rc]).strip()
            if not game or game.lower() == "nan": continue
            key = _norm_key(game)
            console = display_console_name(game_to_console.get(key, "Unknown"))
            weighted_minutes = quarter_minutes * rank_weights[rc]
            console_weights[console] = console_weights.get(console, 0) + weighted_minutes
            period_weights[period] = period_weights.get(period, 0) + weighted_minutes
            game_weights[game] = game_weights.get(game, 0) + weighted_minutes

    for d in [console_weights, period_weights, game_weights]:
        for k in d:
            d[k] /= 1440.0

    total_days = sum(console_weights.values())

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax_console, ax_period, ax_games, ax_genre = axes.flatten()

    # --- Console histogram ---
    consoles = sorted(console_weights.keys(), key=lambda c: console_weights[c], reverse=True)
    vals = [console_weights[c] for c in consoles]
    cols = [color_for_console(c) for c in consoles]
    short_labels = [short_console_name(c) for c in consoles]

    ax_console.bar(short_labels, vals, color=cols, edgecolor="#dddddd", linewidth=1.2)
    ax_console.set_title("By Console", fontsize=13, pad=10)
    ax_console.set_ylabel("Days Played", fontsize=11)
    for label in ax_console.get_xticklabels():
        label.set_rotation(25)
        label.set_fontsize(9)
    ax_console.text(0.5, 0.8, f"Total Lifetime ≈ {total_days:.1f} days",
                    ha="center", va="bottom", transform=ax_console.transAxes,
                    fontsize=11, color="#cccccc")
    ax_console.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax_console.set_ylim(0, max(vals)*1.15)  # <-- Add 15% headroom
    legend_handles = [Patch(facecolor=color_for_console(c), edgecolor="#cccccc",
                            label=f"{short_console_name(c)} ({c})") for c in consoles]
    ax_console.legend(handles=legend_handles, title="Consoles",
                      loc="upper right", frameon=False, fontsize=8)

    # --- Period histogram ---
    ordered_periods = ["PreSchool","Primary School","Secondary School","University","Early Career","Adulthood"]
    periods = [p for p in ordered_periods if p in period_weights]
    vals_p = [period_weights[p] for p in periods]
    cols_p = [PERIOD_COLORS.get(p, _period_fallback[i % len(_period_fallback)]) for i, p in enumerate(periods)]
    ax_period.bar(periods, vals_p, color=cols_p, edgecolor="#dddddd", linewidth=1.2)
    ax_period.set_title("By Life Period", fontsize=13, pad=10)
    ax_period.set_ylabel("Days Played", fontsize=11)
    for label in ax_period.get_xticklabels():
        label.set_rotation(25)
        label.set_fontsize(8.5)
    for i, v in enumerate(vals_p):
        ax_period.text(i, v + max(vals_p)*0.01, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax_period.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax_period.set_ylim(0, max(vals_p)*1.15)

        # --- Top 12 Games histogram (with short labels + legend) ---
    SHORT_NAMES = {
        "Call of Duty Modern Warfare 2": "COD MW2",
        "Call of Duty Modern Warfare 3": "COD MW3",
        "Call of Duty Black Ops": "COD BO",
        "Call of Duty World at War": "COD WAW",
        "Age of Empires II The Age of Kings": "AOE II",
        "Age of Empires Definitive Edition": "AOE DE",
        "World of Warcraft": "WoW",
        "Worms World Party": "Worms WP",
        "RuneScape 2": "RuneScape 2",
        "Overcooked 2": "Overcooked 2",
        "Elden Ring": "Elden Ring",
        "Halo 3": "Halo 3",
    }

    top_games = sorted(game_weights.items(), key=lambda kv: kv[1], reverse=True)[:12]

    # Apply short labels
    game_full_names = [g.strip() for g, _ in top_games]
    game_names = [SHORT_NAMES.get(g, g) for g in game_full_names]
    game_vals = [v for _, v in top_games]
    game_consoles = [display_console_name(game_to_console.get(_norm_key(g), "Unknown")) for g in game_full_names]
    game_cols = [color_for_console(c) for c in game_consoles]
    game_short = [short_console_name(c) for c in game_consoles]

    # Optional: break long short names into two lines
    def break_label(name):
        parts = name.split()
        if len(parts) > 2:
            half = len(parts)//2
            return " ".join(parts[:half]) + "\n" + " ".join(parts[half:])
        return name

    game_names = [break_label(g) for g in game_names]

    # Plot
    ax_games.bar(game_names, game_vals, color=game_cols, edgecolor="#dddddd", linewidth=1.2)
    ax_games.set_title("Top 12 Most Played Games", fontsize=13, pad=10)
    ax_games.set_ylabel("Days Played", fontsize=11)
    for label in ax_games.get_xticklabels():
        label.set_rotation(25)
        label.set_fontsize(8.5)
    for i, (v, abbr) in enumerate(zip(game_vals, game_short)):
        ax_games.text(i, v + max(game_vals)*0.02, f"{v:.1f} d\n({abbr})",
                      ha="center", va="bottom", fontsize=8, color="#e0e0e0")
    ax_games.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax_games.set_ylim(0, max(game_vals)*1.25)

    # Legend showing full names
    legend_labels = [f"{SHORT_NAMES.get(g, g)} → {g}" for g in game_full_names if g in SHORT_NAMES]
    legend_text = "\n".join(legend_labels)
    ax_games.text(0.55, 0.7, legend_text, transform=ax_games.transAxes,
                  va="center", ha="left", fontsize=8, color="#cccccc",
                  bbox=dict(facecolor="#1e1e1e", alpha=0.0, edgecolor="none"))


    # --- Genre histogram ---
    genre_weights = {}
    if "Game" in df_all.columns and "Genre" in df_all.columns:
        game_to_genre = {_norm_key(row["Game"]): str(row["Genre"]).strip()
                         for _, row in df_all.dropna(subset=["Game", "Genre"]).iterrows()}
        for g, days in game_weights.items():
            key = _norm_key(g)
            if key in game_to_genre:
                genre = game_to_genre[key]
                genre_weights[genre] = genre_weights.get(genre, 0) + days
    top_genres = sorted(genre_weights.items(), key=lambda kv: kv[1], reverse=True)[:10]
    genres, vals_g = zip(*top_genres) if top_genres else ([], [])
    GENRE_COLORS = {
        "Platformer": "#1b9e77", "Shooter": "#ff4d4d", "RPG": "#ff8c1a", "MMORPG": "#ffb347",
        "Strategy": "#4daf4a", "Sports": "#e6ab02", "Puzzle": "#984ea3", "Fighting": "#a65628",
        "Stealth": "#999999", "Simulation": "#66c2a5", "Adventure": "#377eb8", "Other": "#bdbdbd"
    }
    def color_for_genre(g):
        for key, color in GENRE_COLORS.items():
            if re.search(key, g, re.IGNORECASE):
                return color
        return "#888888"
    cols_g = [color_for_genre(g) for g in genres]
    ax_genre.bar(genres, vals_g, color=cols_g, edgecolor="#dddddd", linewidth=1.2)
    ax_genre.set_title("By Game Genre", fontsize=13, pad=10)
    ax_genre.set_ylabel("Days Played", fontsize=11)
    for label in ax_genre.get_xticklabels():
        label.set_rotation(25)
        label.set_fontsize(8.5)
    for i, v in enumerate(vals_g):
        ax_genre.text(i, v + max(vals_g)*0.02, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax_genre.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax_genre.set_ylim(0, max(vals_g)*1.15)

    # --- Footer ---
    fig.text(0.01, 0.01, "Source: approximate gaming history", color="#888888", fontsize=8)
    plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.08, hspace=0.4, wspace=0.3)
    # plt.show()
    # --- Save figure to PNG ---
    output_path = os.path.join(script_dir, "gaming_summary.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"✅ Figure saved to: {output_path}")

if __name__ == "__main__":
    main()
