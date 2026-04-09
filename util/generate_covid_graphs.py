"""
Build weekly COVID graph snapshots from OxCGRT data.

Each snapshot is a PyG Data object where:
  - nodes     = countries
  - x         = [daily_cases, daily_deaths, stringency]   (node features)
  - action    = [C1, C2, C6, H6]                          (government policy)
  - y         = x at t+1                                  (prediction target)
  - edge_index = geographic border adjacency

Weekly aggregation:
  - daily_cases / daily_deaths : mean over the 7 days
  - stringency, C1, C2, C6, H6 : mode over the 7 days
"""

import os
import torch
import pandas as pd
import numpy as np
from scipy import stats
from torch_geometric.data import Data

# ── columns ──────────────────────────────────────────────────────────────────
POLICY_COLS = [
    "C1M_combined_numeric",
    "C2M_combined_numeric",
    "C6M_combined_numeric",
    "H6M_combined_numeric",
]
STRINGENCY_COL = "StringencyIndex_Average"
CASES_COL      = "ConfirmedCases"
DEATHS_COL     = "ConfirmedDeaths"

# ── geographic border adjacency (ISO-3 pairs, undirected) ────────────────────
# Each entry is (A, B) meaning A and B share a land/river/strait border.
BORDER_PAIRS = [
    # Africa
    ("DZA","TUN"),("DZA","LBY"),("DZA","NER"),("DZA","MLI"),("DZA","MRT"),("DZA","MAR"),
    ("DZA","ESH"),("MAR","ESH"),("MAR","MRT"),("TUN","LBY"),
    ("LBY","EGY"),("LBY","SDN"),("LBY","TCD"),("LBY","NER"),
    ("EGY","SDN"),("EGY","ISR"),("EGY","PSE"),
    ("SDN","ERI"),("SDN","ETH"),("SDN","SSD"),("SDN","CAF"),("SDN","TCD"),("SDN","LBY"),
    ("SSD","ETH"),("SSD","KEN"),("SSD","UGA"),("SSD","COD"),("SSD","CAF"),
    ("ETH","ERI"),("ETH","DJI"),("ETH","SOM"),("ETH","KEN"),("ETH","SSD"),
    ("ERI","DJI"),
    ("DJI","SOM"),
    ("SOM","KEN"),
    ("KEN","UGA"),("KEN","TZA"),
    ("UGA","COD"),("UGA","RWA"),("UGA","TZA"),
    ("RWA","COD"),("RWA","BDI"),("RWA","TZA"),
    ("BDI","COD"),("BDI","TZA"),
    ("TZA","COD"),("TZA","ZMB"),("TZA","MWI"),("TZA","MOZ"),
    ("COD","CAF"),("COD","COG"),("COD","GAB"),("COD","AGO"),("COD","ZMB"),("COD","SSD"),
    ("COG","CAF"),("COG","GAB"),("COG","AGO"),("COG","CMR"),
    ("CAF","CMR"),("CAF","TCD"),("CAF","SDN"),
    ("TCD","NER"),("TCD","NGA"),("TCD","CMR"),
    ("NER","NGA"),("NER","BEN"),("NER","BFA"),("NER","MLI"),
    ("NGA","BEN"),("NGA","CMR"),
    ("BEN","BFA"),("BEN","TGO"),
    ("BFA","MLI"),("BFA","CIV"),("BFA","GHA"),("BFA","TGO"),
    ("GHA","CIV"),("GHA","TGO"),
    ("TGO","CIV"),
    ("CIV","LBR"),("CIV","GIN"),("CIV","MLI"),
    ("LBR","GIN"),("LBR","SLE"),
    ("GIN","SLE"),("GIN","SEN"),("GIN","MLI"),("GIN","GMB"),
    ("SLE","GIN"),
    ("SEN","GMB"),("SEN","GIN"),("SEN","MLI"),("SEN","MRT"),
    ("GMB","SEN"),
    ("MLI","MRT"),
    ("MRT","ESH"),
    ("CMR","GAB"),("CMR","NGA"),
    ("GAB","GNQ"),
    ("AGO","ZMB"),("AGO","NAM"),("AGO","COG"),
    ("ZMB","ZWE"),("ZMB","MOZ"),("ZMB","MWI"),("ZMB","NAM"),("ZMB","BWA"),
    ("ZWE","MOZ"),("ZWE","BWA"),("ZWE","ZAF"),
    ("MOZ","ZAF"),("MOZ","SWZ"),("MOZ","MWI"),("MOZ","TZA"),
    ("SWZ","ZAF"),
    ("LSO","ZAF"),
    ("NAM","ZAF"),("NAM","BWA"),
    ("BWA","ZAF"),
    # Middle East
    ("ISR","PSE"),("ISR","JOR"),("ISR","LBN"),("ISR","SYR"),
    ("PSE","JOR"),
    ("JOR","SYR"),("JOR","IRQ"),("JOR","SAU"),
    ("LBN","SYR"),
    ("SYR","IRQ"),("SYR","TUR"),
    ("IRQ","IRN"),("IRQ","KWT"),("IRQ","SAU"),("IRQ","TUR"),
    ("SAU","KWT"),("SAU","ARE"),("SAU","QAT"),("SAU","BHR"),("SAU","OMN"),("SAU","YEM"),
    ("ARE","OMN"),("ARE","QAT"),
    ("OMN","YEM"),
    ("IRN","TUR"),("IRN","ARM"),("IRN","AZE"),("IRN","TKM"),("IRN","AFG"),("IRN","PAK"),
    ("YEM","OMN"),
    # Central Asia
    ("TKM","UZB"),("TKM","KAZ"),("TKM","AFG"),
    ("UZB","KAZ"),("UZB","KGZ"),("UZB","TJK"),("UZB","AFG"),
    ("KAZ","RUS"),("KAZ","CHN"),("KAZ","KGZ"),
    ("KGZ","TJK"),("KGZ","CHN"),
    ("TJK","AFG"),("TJK","CHN"),
    ("AFG","PAK"),("AFG","CHN"),
    ("PAK","IND"),("PAK","CHN"),
    # South Asia
    ("IND","CHN"),("IND","NPL"),("IND","BTN"),("IND","BGD"),("IND","MMR"),("IND","LKA"),
    ("NPL","CHN"),
    ("BTN","CHN"),
    ("BGD","MMR"),
    # Southeast / East Asia
    ("CHN","MNG"),("CHN","RUS"),("CHN","PRK"),("CHN","VNM"),("CHN","LAO"),("CHN","MMR"),
    ("MNG","RUS"),
    ("PRK","RUS"),("PRK","KOR"),
    ("VNM","LAO"),("VNM","KHM"),
    ("LAO","THA"),("LAO","KHM"),("LAO","MMR"),
    ("THA","KHM"),("THA","MYS"),("THA","MMR"),
    ("KHM","THA"),
    ("MYS","IDN"),("MYS","BRN"),("MYS","SGP"),
    ("IDN","TLS"),("IDN","PNG"),
    ("PNG","AUS"),  # maritime / Torres Strait proximity
    # Europe
    ("PRT","ESP"),
    ("ESP","FRA"),("ESP","AND"),
    ("FRA","AND"),("FRA","MCO"),("FRA","ITA"),("FRA","CHE"),("FRA","DEU"),("FRA","LUX"),("FRA","BEL"),
    ("ITA","CHE"),("ITA","AUT"),("ITA","SVN"),("ITA","SMR"),("ITA","VAT"),
    ("CHE","AUT"),("CHE","LIE"),("CHE","DEU"),
    ("AUT","DEU"),("AUT","LIE"),("AUT","SVK"),("AUT","HUN"),("AUT","SVN"),("AUT","CZE"),
    ("DEU","LUX"),("DEU","BEL"),("DEU","NLD"),("DEU","DNK"),("DEU","POL"),("DEU","CZE"),
    ("BEL","LUX"),("BEL","NLD"),
    ("NLD","DEU"),
    ("LUX","BEL"),
    ("DNK","SWE"),  # Øresund bridge
    ("SWE","NOR"),("SWE","FIN"),
    ("NOR","FIN"),("NOR","RUS"),
    ("FIN","RUS"),("FIN","EST"),  # maritime proximity
    ("EST","LVA"),("EST","RUS"),
    ("LVA","LTU"),("LVA","BLR"),("LVA","RUS"),
    ("LTU","BLR"),("LTU","POL"),("LTU","RUS"),  # Kaliningrad
    ("POL","CZE"),("POL","SVK"),("POL","UKR"),("POL","BLR"),("POL","RUS"),
    ("CZE","SVK"),("CZE","AUT"),("CZE","DEU"),("CZE","POL"),
    ("SVK","HUN"),("SVK","UKR"),("SVK","AUT"),
    ("HUN","AUT"),("HUN","SVN"),("HUN","HRV"),("HUN","SRB"),("HUN","ROU"),("HUN","UKR"),
    ("SVN","HRV"),("SVN","AUT"),("SVN","ITA"),
    ("HRV","BIH"),("HRV","SRB"),("HRV","MNE"),
    ("BIH","SRB"),("BIH","MNE"),
    ("SRB","MNE"),("SRB","RKS"),("SRB","MKD"),("SRB","BGR"),("SRB","ROU"),
    ("RKS","MKD"),("RKS","ALB"),("RKS","MNE"),
    ("MKD","BGR"),("MKD","GRC"),("MKD","ALB"),
    ("ALB","GRC"),("ALB","MNE"),
    ("GRC","BGR"),("GRC","TUR"),
    ("TUR","BGR"),("TUR","GRC"),("TUR","ARM"),("TUR","GEO"),
    ("ARM","GEO"),("ARM","AZE"),("ARM","IRN"),
    ("AZE","GEO"),("AZE","RUS"),("AZE","IRN"),
    ("GEO","RUS"),
    ("ROU","BGR"),("ROU","MDA"),("ROU","UKR"),("ROU","HUN"),("ROU","SRB"),
    ("MDA","UKR"),
    ("UKR","BLR"),("UKR","RUS"),
    ("BLR","RUS"),
    ("RUS","KAZ"),("RUS","MNG"),("RUS","CHN"),("RUS","PRK"),
    # Americas
    ("CAN","USA"),
    ("USA","MEX"),
    ("MEX","GTM"),("MEX","BLZ"),
    ("GTM","BLZ"),("GTM","HND"),("GTM","SLV"),
    ("BLZ","HND"),
    ("HND","NIC"),("HND","SLV"),
    ("NIC","CRI"),
    ("CRI","PAN"),
    ("PAN","COL"),
    ("COL","VEN"),("COL","BRA"),("COL","PER"),("COL","ECU"),
    ("VEN","BRA"),("VEN","GUY"),
    ("GUY","BRA"),("GUY","SUR"),
    ("SUR","BRA"),("SUR","GUY"),
    ("BRA","PER"),("BRA","BOL"),("BRA","PRY"),("BRA","ARG"),("BRA","URY"),
    ("PER","BOL"),("PER","CHL"),("PER","ECU"),
    ("BOL","CHL"),("BOL","ARG"),("BOL","PRY"),
    ("CHL","ARG"),
    ("ARG","PRY"),("ARG","URY"),
    ("HTI","DOM"),
    # Oceania / island neighbours — no land borders beyond above
]


def build_border_edge_index(country_codes: list[str]) -> torch.Tensor:
    """Build edge_index from BORDER_PAIRS for the given country code list."""
    code2idx = {c: i for i, c in enumerate(country_codes)}
    src, dst = [], []
    for a, b in BORDER_PAIRS:
        if a in code2idx and b in code2idx:
            i, j = code2idx[a], code2idx[b]
            src += [i, j]   # undirected
            dst += [j, i]
    return torch.tensor([src, dst], dtype=torch.long)


# ── helpers ───────────────────────────────────────────────────────────────────

def mode(x):
    result = stats.mode(x, keepdims=True)
    return float(result.mode[0])


def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df["Jurisdiction"] == "NAT_TOTAL"].copy()
    df["date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    num_cols = [CASES_COL, DEATHS_COL, STRINGENCY_COL] + POLICY_COLS
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values(["CountryName", "date"]).reset_index(drop=True)
    return df


def compute_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["daily_cases"]  = df.groupby("CountryName")[CASES_COL].diff().clip(lower=0)
    df["daily_deaths"] = df.groupby("CountryName")[DEATHS_COL].diff().clip(lower=0)
    df = df.dropna(subset=["daily_cases", "daily_deaths"])
    return df


def weekly_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year_week"] = df["date"].dt.to_period("W")
    agg_mean = (
        df.groupby(["CountryName", "CountryCode", "year_week"])
        [["daily_cases", "daily_deaths", STRINGENCY_COL]]
        .mean()
    )
    agg_mode = (
        df.groupby(["CountryName", "CountryCode", "year_week"])[POLICY_COLS]
        .agg(mode)
    )
    weekly = pd.concat([agg_mean, agg_mode], axis=1).reset_index()
    weekly = weekly.rename(columns={
        STRINGENCY_COL: "stringency",
        "C1M_combined_numeric": "C1",
        "C2M_combined_numeric": "C2",
        "C6M_combined_numeric": "C6",
        "H6M_combined_numeric": "H6",
    })
    weekly = weekly.sort_values(["CountryName", "year_week"]).reset_index(drop=True)
    return weekly


def build_snapshots(weekly: pd.DataFrame):
    countries = sorted(weekly["CountryName"].unique())
    # map country name → ISO-3 code
    name2code = (
        weekly[["CountryName", "CountryCode"]]
        .drop_duplicates()
        .set_index("CountryName")["CountryCode"]
        .to_dict()
    )

    weeks = sorted(weekly["year_week"].unique())
    snapshots = []

    for w_idx in range(len(weeks) - 1):
        w_t  = weeks[w_idx]
        w_t1 = weeks[w_idx + 1]

        frame_t  = weekly[weekly["year_week"] == w_t].set_index("CountryName")
        frame_t1 = weekly[weekly["year_week"] == w_t1].set_index("CountryName")

        common = [c for c in countries if c in frame_t.index and c in frame_t1.index]
        if len(common) < 2:
            continue

        codes = [name2code[c] for c in common]
        edge_index = build_border_edge_index(codes)

        node_feat = torch.tensor(
            frame_t.loc[common, ["daily_cases", "daily_deaths", "stringency"]].values,
            dtype=torch.float,
        )
        action = torch.tensor(
            frame_t.loc[common, ["C1", "C2", "C6", "H6"]].values,
            dtype=torch.float,
        )
        target = torch.tensor(
            frame_t1.loc[common, ["daily_cases", "daily_deaths", "stringency"]].values,
            dtype=torch.float,
        )

        data = Data(
            x=node_feat,
            edge_index=edge_index,
            action=action,
            y=target,
            week=str(w_t),
            countries=common,
            country_codes=codes,
            num_nodes=len(common),
        )
        snapshots.append(data)

    return snapshots


def main():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "OxCGRT_simplified_v1.csv")
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "covid_graphs.pt")

    print("Loading data …")
    df = load_and_clean(csv_path)
    print(f"  {len(df):,} national rows, {df['CountryName'].nunique()} countries")

    print("Computing daily cases/deaths …")
    df = compute_daily(df)

    print("Aggregating to weekly …")
    weekly = weekly_aggregate(df)
    print(f"  {weekly['year_week'].nunique()} weeks")

    print("Building graph snapshots …")
    snapshots = build_snapshots(weekly)
    print(f"  {len(snapshots)} snapshots")

    if snapshots:
        s = snapshots[0]
        n_edges = s.edge_index.shape[1] // 2
        print(f"\nExample snapshot (week {s.week}):")
        print(f"  nodes      : {s.num_nodes} countries")
        print(f"  edges      : {n_edges} border pairs ({s.edge_index.shape[1]} directed)")
        print(f"  x shape    : {s.x.shape}  — [daily_cases, daily_deaths, stringency]")
        print(f"  action     : {s.action.shape}  — [C1, C2, C6, H6]")
        print(f"  y shape    : {s.y.shape}  — target node features at t+1")

    torch.save(snapshots, out_path)
    print(f"\nSaved {len(snapshots)} snapshots → {out_path}")


if __name__ == "__main__":
    main()
