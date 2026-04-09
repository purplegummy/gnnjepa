"""
Visualize JEPA encoder embeddings via UMAP.

Produces 3 plots, each coloring countries by:
  1. Geographic region
  2. Stringency level (low / medium / high)
  3. Case level (low / medium / high)

Each point = one country at one week.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import umap

from model.encoder import Encoder
from model.predictor import Predictor
from model.jepa import JEPA


# ── geographic region lookup (ISO-3) ─────────────────────────────────────────
REGIONS = {
    # Europe
    "ALB","AND","AUT","BEL","BIH","BGR","BLR","CHE","CYP","CZE","DEU","DNK",
    "ESP","EST","FIN","FRA","GBR","GRC","HRV","HUN","IRL","ISL","ITA","LIE",
    "LTU","LUX","LVA","MCO","MDA","MKD","MLT","MNE","NLD","NOR","POL","PRT",
    "RKS","ROU","RUS","SMR","SRB","SVK","SVN","SWE","UKR","FRO",
    # North America & Caribbean
    "CAN","USA","MEX","CUB","DOM","HTI","JAM","BHS","BRB","TTO","DMA","PRI",
    "ABW","BMU","GUM","VIR",
    # Central America
    "BLZ","CRI","GTM","HND","NIC","PAN","SLV",
    # South America
    "ARG","BOL","BRA","CHL","COL","ECU","GUY","PER","PRY","SUR","URY","VEN",
    # Middle East
    "ARE","BHR","IRQ","IRN","ISR","JOR","KWT","LBN","OMN","PSE","QAT","SAU",
    "SYR","TUR","YEM",
    # Central Asia
    "KAZ","KGZ","TJK","TKM","UZB","MNG","AFG",
    # South Asia
    "BGD","BTN","IND","LKA","MDV","NPL","PAK",
    # East & Southeast Asia
    "BRN","CHN","HKG","IDN","JPN","KHM","KOR","LAO","MAC","MMR","MYS","PHL",
    "PRK","SGP","THA","TLS","TWN","VNM",
    # Oceania
    "AUS","FJI","KIR","NZL","PNG","SLB","TON","VUT",
    # Africa
    "AGO","BDI","BEN","BFA","BWA","CAF","CMR","COD","COG","CIV","DJI","DZA",
    "EGY","ERI","ETH","GAB","GHA","GIN","GMB","GNQ","KEN","LBR","LBY","LSO",
    "MAR","MDG","MLI","MOZ","MRT","MUS","MWI","NAM","NER","NGA","RWA","SDN",
    "SEN","SLE","SOM","SSD","SWZ","TCD","TGO","TUN","TZA","UGA","ZAF","ZMB",
    "ZWE","CPV","SYC","GRL",
}

REGION_MAP = {}
for code in ["ALB","AND","AUT","BEL","BIH","BGR","BLR","CHE","CYP","CZE","DEU","DNK",
             "ESP","EST","FIN","FRA","GBR","GRC","HRV","HUN","IRL","ISL","ITA","LIE",
             "LTU","LUX","LVA","MCO","MDA","MKD","MLT","MNE","NLD","NOR","POL","PRT",
             "RKS","ROU","RUS","SMR","SRB","SVK","SVN","SWE","UKR","FRO"]:
    REGION_MAP[code] = "Europe"
for code in ["CAN","USA","MEX","CUB","DOM","HTI","JAM","BHS","BRB","TTO","DMA","PRI","ABW","BMU","GUM","VIR"]:
    REGION_MAP[code] = "N. America & Caribbean"
for code in ["BLZ","CRI","GTM","HND","NIC","PAN","SLV"]:
    REGION_MAP[code] = "Central America"
for code in ["ARG","BOL","BRA","CHL","COL","ECU","GUY","PER","PRY","SUR","URY","VEN"]:
    REGION_MAP[code] = "South America"
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
    "Europe":                 "#4e79a7",
    "N. America & Caribbean": "#f28e2b",
    "Central America":        "#e15759",
    "South America":          "#76b7b2",
    "Middle East":            "#59a14f",
    "Central Asia":           "#edc948",
    "South Asia":             "#b07aa1",
    "East/SE Asia":           "#ff9da7",
    "Oceania":                "#9c755f",
    "Africa":                 "#bab0ac",
}


def extract_embeddings(model, snapshots, device):
    """Return embeddings, metadata arrays across all snapshots."""
    model.eval()
    all_z, all_codes, all_stringency, all_cases, all_weeks = [], [], [], [], []

    with torch.no_grad():
        for snap in snapshots:
            x          = snap.x.to(device)
            edge_index = snap.edge_index.to(device)
            z          = model.encoder(x, edge_index).cpu().numpy()  # [N, embed_dim]

            all_z.append(z)
            all_codes.extend(snap.country_codes)
            all_stringency.extend(x[:, 2].cpu().numpy())  # stringency is index 2
            all_cases.extend(x[:, 0].cpu().numpy())       # daily_cases is index 0
            all_weeks.extend([snap.week] * len(snap.country_codes))

    return (
        np.vstack(all_z),
        np.array(all_codes),
        np.array(all_stringency),
        np.array(all_cases),
        np.array(all_weeks),
    )


def plot_by_region(ax, umap_2d, codes):
    regions = np.array([REGION_MAP.get(c, "Other") for c in codes])
    unique  = sorted(set(regions))
    for region in unique:
        mask  = regions == region
        color = REGION_COLORS.get(region, "#aaaaaa")
        ax.scatter(umap_2d[mask, 0], umap_2d[mask, 1],
                   c=color, s=4, alpha=0.5, label=region)
    ax.legend(fontsize=6, markerscale=2, loc="best")
    ax.set_title("Colored by Geographic Region")


def plot_by_continuous(ax, umap_2d, values, label, cmap="viridis"):
    sc = ax.scatter(umap_2d[:, 0], umap_2d[:, 1],
                    c=values, cmap=cmap, s=4, alpha=0.5)
    plt.colorbar(sc, ax=ax, label=label)
    ax.set_title(f"Colored by {label} (normalized)")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder   = Encoder(in_dim=3, hidden_dim=32, out_dim=16)
    predictor = Predictor(in_dim=20, hidden_dim=32, out_dim=16)
    model     = JEPA(encoder, predictor).to(device)
    model.load_state_dict(torch.load('model_weights.pt', map_location=device))

    snapshots = torch.load('data/covid_graphs.pt', weights_only=False)

    print("Extracting embeddings …")
    z, codes, stringency, cases, weeks = extract_embeddings(model, snapshots, device)
    print(f"  {z.shape[0]} embeddings (countries × weeks), dim={z.shape[1]}")

    print("Running UMAP …")
    reducer  = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    umap_2d  = reducer.fit_transform(z)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("JEPA Encoder Embeddings (UMAP)", fontsize=13)

    plot_by_region(axes[0], umap_2d, codes)
    plot_by_continuous(axes[1], umap_2d, stringency, "Stringency", cmap="RdYlGn_r")
    plot_by_continuous(axes[2], umap_2d, cases,      "Daily Cases", cmap="YlOrRd")

    for ax in axes:
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    out = "embeddings_umap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.show()


if __name__ == "__main__":
    main()
