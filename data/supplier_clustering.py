"""
HDBSCAN supplier clustering.

Uses sklearn.cluster.HDBSCAN (available since scikit-learn 1.3) on four numeric
KPIs from supplier_enrichment.py:

  - lead_time_days           (higher = worse)
  - on_time_delivery_pct     (inverted: 100 - OTD, so higher = worse)
  - quality_rejection_pct    (higher = worse)
  - spend_ytd_usd            (log-scaled to reduce skew, kept directionally neutral)

All features are StandardScaler-normalised before clustering so no single KPI
dominates due to magnitude.

HDBSCAN advantages over k-means here:
  - Does not require specifying k upfront
  - Labels genuine outliers as noise (-1) rather than forcing them into a cluster
  - Robust to non-spherical cluster shapes
  - Stable with small datasets when min_cluster_size=2, min_samples=1
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import HDBSCAN

from data.supplier_enrichment import SUPPLIER_METRICS

# Ordinal encoding for risk_level (used only in narrative generation, not as a feature
# to avoid double-counting information already captured in OTD/rejection).
_RISK_ORDER = {"Very Low": 0, "Low": 1, "Medium": 2, "Medium-High": 3, "High": 4}


def _build_feature_matrix():
    ids, names, rows = [], [], []
    for sid, m in SUPPLIER_METRICS.items():
        ids.append(sid)
        names.append(m.get("name", sid))
        rows.append([
            float(m.get("lead_time_days", 30)),
            100.0 - float(m.get("on_time_delivery_pct", 80)),   # invert: higher = worse
            float(m.get("quality_rejection_pct", 2.0)),
            float(np.log1p(m.get("spend_ytd_usd", 0))),         # log scale
        ])
    return ids, names, np.array(rows, dtype=float)


def _auto_label(avg_otd: float, avg_rej: float, avg_lead: float) -> str:
    if avg_otd >= 92 and avg_rej <= 1.2:
        return "Top Performers"
    if avg_otd < 77 or avg_rej >= 4.0:
        return "High Risk"
    if avg_otd >= 84 and avg_lead <= 15:
        return "Reliable Mid-Tier"
    return "Moderate Risk"


def _cluster_narrative(label: str, avg_otd: float, avg_rej: float, avg_lead: float) -> str:
    if label == "Top Performers":
        return (
            f"{avg_otd}% on-time delivery and {avg_rej}% quality rejection rate. "
            "Recommended for critical, high-volume, or time-sensitive procurement. "
            "Candidates for preferred-supplier agreements and volume growth."
        )
    if label == "High Risk":
        return (
            f"{avg_otd}% on-time delivery and {avg_rej}% quality rejection rate. "
            "Active monitoring and corrective action plans required. "
            "Evaluate alternative or dual-source options to reduce dependency."
        )
    if label == "Reliable Mid-Tier":
        return (
            f"{avg_otd}% OTD with {avg_lead}-day average lead time. "
            "Solid performers suitable for expanded sourcing to reduce concentration risk "
            "on top-tier suppliers."
        )
    return (
        f"{avg_otd}% OTD and {avg_rej}% rejection rate. "
        "Performance is within acceptable range but warrants supplier development "
        "engagement to drive consistent improvement."
    )


def run_clustering() -> dict:
    """
    Run HDBSCAN on supplier KPIs.

    Returns a dict with keys:
      clusters         — list of cluster dicts (id, label, members, avg KPIs, spend)
      noise_suppliers  — suppliers HDBSCAN could not assign to any cluster
      total_suppliers  — int
      method           — "HDBSCAN"
      features         — list of feature names used
    """
    ids, names, X = _build_feature_matrix()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clusterer = HDBSCAN(min_cluster_size=2, min_samples=1, metric="euclidean")
    labels = clusterer.fit_predict(X_scaled)

    # Group suppliers by cluster label
    bucket: dict[int, list] = {}
    noise: list = []
    for i, (sid, name, cluster_id) in enumerate(zip(ids, names, labels)):
        m = SUPPLIER_METRICS[sid]
        entry = {
            "supplier_id": sid,
            "name": name,
            "on_time_delivery_pct": m.get("on_time_delivery_pct"),
            "quality_rejection_pct": m.get("quality_rejection_pct"),
            "lead_time_days": m.get("lead_time_days"),
            "spend_ytd_usd": m.get("spend_ytd_usd"),
            "risk_level": m.get("risk_level"),
            "country": m.get("country"),
        }
        if cluster_id == -1:
            noise.append(entry)
        else:
            bucket.setdefault(int(cluster_id), []).append(entry)

    # Build labelled cluster summaries
    result_clusters = []
    for cluster_id, members in sorted(bucket.items()):
        metrics = [SUPPLIER_METRICS[m["supplier_id"]] for m in members]
        avg_otd   = round(float(np.mean([m.get("on_time_delivery_pct", 80) for m in metrics])), 1)
        avg_rej   = round(float(np.mean([m.get("quality_rejection_pct", 2)  for m in metrics])), 1)
        avg_lead  = round(float(np.mean([m.get("lead_time_days", 30)        for m in metrics])), 1)
        total_sp  = int(sum(m.get("spend_ytd_usd", 0) for m in metrics))
        label     = _auto_label(avg_otd, avg_rej, avg_lead)

        result_clusters.append({
            "id": cluster_id,
            "label": label,
            "size": len(members),
            "members": members,
            "avg_otd_pct": avg_otd,
            "avg_rejection_pct": avg_rej,
            "avg_lead_time_days": avg_lead,
            "total_spend_ytd_usd": total_sp,
            "narrative": _cluster_narrative(label, avg_otd, avg_rej, avg_lead),
        })

    return {
        "clusters": result_clusters,
        "noise_suppliers": noise,
        "total_suppliers": len(ids),
        "method": "HDBSCAN",
        "features": [
            "lead_time_days",
            "on_time_delivery_pct",
            "quality_rejection_pct",
            "spend_ytd_usd (log-scaled)",
        ],
    }


def cluster_documents() -> list:
    """
    Generate text documents from clustering results for indexing into pgvector.
    Called by aria_graph._all_seed_documents() on reindex.
    """
    result = run_clustering()
    docs = []

    for c in result["clusters"]:
        member_names = ", ".join(m["name"] for m in c["members"])
        docs.append(
            f"Supplier Cluster Analysis: {c['label']}\n"
            f"Cluster ID: {c['id']}\n"
            f"Members ({c['size']} suppliers): {member_names}\n"
            f"Avg On-Time Delivery: {c['avg_otd_pct']}%\n"
            f"Avg Quality Rejection Rate: {c['avg_rejection_pct']}%\n"
            f"Avg Lead Time: {c['avg_lead_time_days']} days\n"
            f"Total YTD Spend: ${c['total_spend_ytd_usd']:,}\n"
            f"Assessment: {c['narrative']}"
        )

    if result["noise_suppliers"]:
        noise_names = ", ".join(s["name"] for s in result["noise_suppliers"])
        docs.append(
            f"Supplier Cluster Analysis: Outliers\n"
            f"Members: {noise_names}\n"
            f"Assessment: These suppliers have a distinct KPI profile that does not "
            f"closely match any peer group. Each should be evaluated individually "
            f"against category-specific benchmarks."
        )

    # Add a summary document covering the whole supplier base segmentation
    cluster_lines = "\n".join(
        f"  - {c['label']}: {c['size']} suppliers (avg OTD {c['avg_otd_pct']}%, "
        f"avg rejection {c['avg_rejection_pct']}%, total spend ${c['total_spend_ytd_usd']:,})"
        for c in result["clusters"]
    )
    docs.append(
        f"Supplier Base Segmentation Summary\n"
        f"Total suppliers analysed: {result['total_suppliers']}\n"
        f"Clustering method: HDBSCAN (density-based, no fixed k)\n"
        f"Clusters identified:\n{cluster_lines}\n"
        f"Noise / outliers: {len(result['noise_suppliers'])} supplier(s)\n"
        f"Features used: {', '.join(result['features'])}"
    )

    return docs


if __name__ == "__main__":
    result = run_clustering()
    print(f"\nHDBSCAN clustering — {result['total_suppliers']} suppliers\n")
    for c in result["clusters"]:
        names = [m["name"] for m in c["members"]]
        print(f"  Cluster {c['id']} [{c['label']}] — {c['size']} suppliers")
        print(f"    Members   : {', '.join(names)}")
        print(f"    Avg OTD   : {c['avg_otd_pct']}%")
        print(f"    Avg Rej   : {c['avg_rejection_pct']}%")
        print(f"    Avg Lead  : {c['avg_lead_time_days']} days")
        print(f"    Spend YTD : ${c['total_spend_ytd_usd']:,}\n")
    if result["noise_suppliers"]:
        print(f"  Outliers: {', '.join(s['name'] for s in result['noise_suppliers'])}")
