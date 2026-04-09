"""
Two visualizations of JEPA encoder embeddings:

  Plot 1 — Single week snapshot: one point per country, colored by region /
            stringency / cases. Fit UMAP on that week only.

  Plot 2 — Country trajectories: 4 chosen countries, all timesteps shown as a
            line through embedding space. UMAP fit on all timesteps of those 4
            countries, colored by week index.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import umap

from model.encoder import Encoder
from model.predictor import Predictor
from model.jepa import JEPA


# ── settings ─────────────────────────────────────────────────────────────────
SNAPSHOT_WEEK   = "2021-01-04/2021-01-10"   # change to any week in the data
TRAJECTORY_COUNTRIES = ["USA", "IND", "BRA", "DEU"]  # ISO-3 codes

# ── region map ───────────────────────────────────────────────────────────────
REGION_MAP = {}
for code in ["ALB","AND","AUT","BEL","BIH","BGR","BLR","CHE","CYP","CZE","DEU","DNK",
             "ESP","EST","FIN","FRA","GBR","GRC","HRV","HUN","IRL","ISL","ITA","LIE",
             "LTU","LUX","LVA","MCO","MDA","MKD","MLT","MNE","NLD","NOR","POL","PRT",
             "RKS","ROU","RUS","SMR","SRB","SVK","SVN","SWE","UKR","FRO"]:
    REGION_MAP[code] = "Europe"
for code in ["CAN","USA","MEX","CUB","DOM","HTI","JAM","BHS","BRB","TTO","DMA","PRI","ABW","BMU","GUM","VIR"]:
    REGION_MAP[code] = "N. America"
for code in ["BLZ","CRI","GTM","HND","NIC","PAN","SLV"]:
    REGION_MAP[code] = "C. America"
for code in ["ARG","BOL","BRA","CHL","COL","ECU","GUY","PER","PRY","SUR","URY","VEN"]:
    REGION_MAP[code] = "S. America"
for code in ["ARE","BHR","IRQ","IRN","ISR","JOR","KWT","LBN","OMN","PSE","QAT","SAU","SYR","TUR","YEM"]:
    REGION_MAP[code] = "Middle East"
for code in ["KAZ","KGZ","TJK","TKM","UZB","MNG","AFG"]:
    REGION_MAP[code] = "Central Asia"
for code in ["BGD","BTN","IND","LKA","NPL","PAK"]:
    REGION_MAP[code] = "South Asia"
for code in ["BRN","CHN","HKG","IDN","JPN","KHM","KOR","LAO","MAC","MMR","MYS","PHL","PRK","SGP","THA","TLS","TWN","VNM"]:
    REGION_MAP[code] = "East/SE Asia"
for code in ["AUS","FJI","KIR","NZL","PNG","SLB","TON","VUT"]:
    REGION_MAP[code] = "Oceania"
for code in ["AGO","BDI","BEN","BFA","BWA","CAF","CMR","COD","COG","CIV","DJI","DZA",
             "EGY","ERI","ETH","GAB","GHA","GIN","GMB","KEN","LBR","LBY","LSO",
             "MAR","MDG","MLI","MOZ","MRT","MUS","MWI","NAM","NER","NGA","RWA","SDN",
             "SEN","SLE","SOM","SSD","SWZ","TCD","TGO","TUN","TZA","UGA","ZAF","ZMB",
             "ZWE","CPV","SYC","GRL"]:
    REGION_MAP[code] = "Africa"

REGION_COLORS = {
    "Europe":       "#4e79a7", "N. America":   "#f28e2b",
    "C. America":   "#e15759", "S. America":   "#76b7b2",
    "Middle East":  "#59a14f", "Central Asia":  "#edc948",
    "South Asia":   "#b07aa1", "East/SE Asia":  "#ff9da7",
    "Oceania":      "#9c755f", "Africa":        "#bab0ac",
}

TRAJ_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]


# ── helpers ───────────────────────────────────────────────────────────────────
def get_encoder(device):
    encoder   = Encoder(in_dim=3, hidden_dim=32, out_dim=16)
    predictor = Predictor(in_dim=20, hidden_dim=32, out_dim=16)
    model     = JEPA(encoder, predictor).to(device)
    model.load_state_dict(torch.load('model_weights.pt', map_location=device))
    model.eval()
    return model


def encode_snapshot(model, snap, device):
    with torch.no_grad():
        z = model.encoder(snap.x.to(device), snap.edge_index.to(device)).cpu().numpy()
    return z


# ── Plot 1: single week ───────────────────────────────────────────────────────
def plot_single_week(model, snapshots, device):
    # find the snapshot for the chosen week
    snap = next((s for s in snapshots if s.week == SNAPSHOT_WEEK), None)
    if snap is None:
        available = [s.week for s in snapshots]
        snap = snapshots[len(snapshots) // 2]  # fall back to middle week
        print(f"  Week '{SNAPSHOT_WEEK}' not found, using '{snap.week}' instead.")
        print(f"  Available weeks: {available[:5]} … {available[-5:]}")

    print(f"  Encoding week: {snap.week}  ({snap.num_nodes} countries)")
    z = encode_snapshot(model, snap, device)

    reducer = umap.UMAP(n_neighbors=10, min_dist=0.2, random_state=42)
    umap_2d = reducer.fit_transform(z)

    codes      = snap.country_codes
    stringency = snap.x[:, 2].numpy()
    cases      = snap.x[:, 0].numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"JEPA Embeddings — week of {snap.week}", fontsize=13)

    # region
    ax = axes[0]
    regions = [REGION_MAP.get(c, "Other") for c in codes]
    for region in sorted(set(regions)):
        mask  = np.array(regions) == region
        color = REGION_COLORS.get(region, "#aaaaaa")
        ax.scatter(umap_2d[mask, 0], umap_2d[mask, 1], c=color, s=30, alpha=0.8, label=region)
    ax.legend(fontsize=6, markerscale=1.5)
    ax.set_title("By Region")

    # stringency
    ax = axes[1]
    sc = ax.scatter(umap_2d[:, 0], umap_2d[:, 1], c=stringency, cmap="RdYlGn_r", s=30, alpha=0.8)
    plt.colorbar(sc, ax=ax, label="Stringency (normalized)")
    ax.set_title("By Stringency")

    # cases
    ax = axes[2]
    sc = ax.scatter(umap_2d[:, 0], umap_2d[:, 1], c=cases, cmap="YlOrRd", s=30, alpha=0.8)
    plt.colorbar(sc, ax=ax, label="Daily Cases (normalized)")
    ax.set_title("By Daily Cases")

    # label a few notable countries
    notable = {"USA", "CHN", "IND", "BRA", "DEU", "GBR", "FRA", "ZAF", "NGA", "AUS"}
    for i, code in enumerate(codes):
        if code in notable:
            axes[0].annotate(code, (umap_2d[i, 0], umap_2d[i, 1]), fontsize=6, alpha=0.9)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")

    plt.tight_layout()
    plt.savefig("viz_single_week.png", dpi=150, bbox_inches="tight")
    print("  Saved → viz_single_week.png")
    plt.show()


# ── Plot 2: country trajectories ──────────────────────────────────────────────
def plot_trajectories(model, snapshots, device):
    # collect embeddings for chosen countries across all weeks
    country_zs = {c: [] for c in TRAJECTORY_COUNTRIES}
    weeks = []

    for snap in snapshots:
        code_list = snap.country_codes
        z = encode_snapshot(model, snap, device)
        for country in TRAJECTORY_COUNTRIES:
            if country in code_list:
                idx = code_list.index(country)
                country_zs[country].append(z[idx])
        weeks.append(snap.week)

    # stack all embeddings for joint UMAP fit
    all_z, all_labels = [], []
    for country in TRAJECTORY_COUNTRIES:
        zs = np.array(country_zs[country])
        all_z.append(zs)
        all_labels.extend([country] * len(zs))

    all_z = np.vstack(all_z)
    print(f"  Fitting UMAP on {len(all_z)} embeddings ({len(TRAJECTORY_COUNTRIES)} countries × weeks)")

    reducer = umap.UMAP(n_neighbors=10, min_dist=0.3, random_state=42)
    umap_2d = reducer.fit_transform(all_z)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_title(f"Country Embedding Trajectories Over Time\n({', '.join(TRAJECTORY_COUNTRIES)})", fontsize=12)

    ptr = 0
    n_weeks = len(snapshots)
    for i, country in enumerate(TRAJECTORY_COUNTRIES):
        n = len(country_zs[country])
        pts = umap_2d[ptr:ptr + n]
        color = TRAJ_COLORS[i]
        cmap  = cm.get_cmap("Blues" if i == 0 else
                            "Oranges" if i == 1 else
                            "Greens" if i == 2 else "Purples")

        # draw line
        ax.plot(pts[:, 0], pts[:, 1], c=color, alpha=0.4, linewidth=1)

        # color points by time
        week_indices = np.linspace(0, 1, n)
        for j in range(n):
            ax.scatter(pts[j, 0], pts[j, 1], color=cmap(0.3 + 0.6 * week_indices[j]),
                       s=20, zorder=3)

        # mark start and end
        ax.scatter(*pts[0],  marker="^", s=80, color=color, zorder=5, label=f"{country} start")
        ax.scatter(*pts[-1], marker="s", s=80, color=color, zorder=5, label=f"{country} end")
        ax.annotate(f"{country}", pts[0], fontsize=9, fontweight="bold", color=color)

        ptr += n

    ax.legend(fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")

    plt.tight_layout()
    plt.savefig("viz_trajectories.png", dpi=150, bbox_inches="tight")
    print("  Saved → viz_trajectories.png")
    plt.show()


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model     = get_encoder(device)
    snapshots = torch.load('data/covid_graphs.pt', weights_only=False)

    print("=== Plot 1: Single week snapshot ===")
    plot_single_week(model, snapshots, device)

    print("\n=== Plot 2: Country trajectories ===")
    plot_trajectories(model, snapshots, device)


if __name__ == "__main__":
    main()
