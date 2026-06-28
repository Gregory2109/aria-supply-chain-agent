"""
Supplier performance enrichment — keyed by SAP Supplier ID.

These metrics augment the basic master data returned by API_BUSINESS_PARTNER,
which only carries BP ID, name, and country. Lead times, on-time delivery rates,
risk scores, and qualitative notes come from this layer.

In a real deployment these would come from SAP Ariba, a vendor scorecard system,
or a procurement analytics platform. For this sandbox they are realistic figures
aligned to the actual supplier IDs present in the SAP API Business Hub dataset.
"""

# Dict[supplier_id -> dict of performance metrics]
SUPPLIER_METRICS = {
    "17300001": {
        "name": "Domestic US Supplier 1",
        "country": "US",
        "lead_time_days": 2,
        "on_time_delivery_pct": 97,
        "quality_rejection_pct": 0.5,
        "risk_level": "Very Low",
        "payment_terms": "Net 30",
        "spend_ytd_usd": 1_240_000,
        "recent_issues": "None reported.",
        "notes": "Primary domestic supplier. Consistent top performer across all KPIs.",
    },
    "17300002": {
        "name": "Domestic US Supplier 2",
        "country": "US",
        "lead_time_days": 5,
        "on_time_delivery_pct": 88,
        "quality_rejection_pct": 1.8,
        "risk_level": "Medium",
        "payment_terms": "Net 30",
        "spend_ytd_usd": 680_000,
        "recent_issues": "Two late deliveries in Q2 2026 due to regional freight disruption.",
        "notes": "Generally reliable but showing signs of capacity strain in H1 2026.",
    },
    "17300080": {
        "name": "Domestic US Supplier (Ariba Network)",
        "country": "US",
        "lead_time_days": 3,
        "on_time_delivery_pct": 91,
        "quality_rejection_pct": 1.1,
        "risk_level": "Low",
        "payment_terms": "Net 45",
        "spend_ytd_usd": 420_000,
        "recent_issues": "None reported.",
        "notes": "Onboarded via SAP Ariba Network. Strong compliance record.",
    },
    "USSU-VSF01": {
        "name": "CyclePartsOnly",
        "country": "US",
        "lead_time_days": 21,
        "on_time_delivery_pct": 85,
        "quality_rejection_pct": 2.4,
        "risk_level": "Medium",
        "payment_terms": "Net 45",
        "spend_ytd_usd": 310_000,
        "recent_issues": "Lead time increased 15% in Q1 2026 due to raw material shortages.",
        "notes": "Specialist parts supplier. Delivery performance trending down over 2 quarters.",
    },
    "USSU-VSF04": {
        "name": "Road'sEndParts",
        "country": "US",
        "lead_time_days": 30,
        "on_time_delivery_pct": 75,
        "quality_rejection_pct": 3.9,
        "risk_level": "High",
        "payment_terms": "Net 60",
        "spend_ytd_usd": 195_000,
        "recent_issues": "Recurring logistics delays. Two quality holds in 2026. Under formal review.",
        "notes": "On supplier improvement plan since May 2026. Alternative supplier evaluation in progress.",
    },
    "USSU-VSF05": {
        "name": "YourTurnParts",
        "country": "US",
        "lead_time_days": 14,
        "on_time_delivery_pct": 78,
        "quality_rejection_pct": 2.9,
        "risk_level": "Medium-High",
        "payment_terms": "Net 45",
        "spend_ytd_usd": 260_000,
        "recent_issues": "Notified 12% price increase effective Q3 2026. On-time delivery declining.",
        "notes": "Price increase under negotiation. Sourcing team evaluating dual-source strategy.",
    },
    "USSU-VSF06": {
        "name": "SteadyParts",
        "country": "US",
        "lead_time_days": 10,
        "on_time_delivery_pct": 86,
        "quality_rejection_pct": 1.5,
        "risk_level": "Medium",
        "payment_terms": "Net 30",
        "spend_ytd_usd": 375_000,
        "recent_issues": "Minor inventory discrepancy reported in March 2026, resolved.",
        "notes": "Steady mid-tier supplier. Consistent but not exceptional across KPIs.",
    },
    "USSU-VSF07": {
        "name": "Mission Possible Parts",
        "country": "US",
        "lead_time_days": 21,
        "on_time_delivery_pct": 80,
        "quality_rejection_pct": 2.1,
        "risk_level": "Medium",
        "payment_terms": "Net 45",
        "spend_ytd_usd": 285_000,
        "recent_issues": "Workforce strike risk flagged in May 2026. Monitoring ongoing.",
        "notes": "Labor relations risk is the primary concern. Contingency stock buffer recommended.",
    },
    "USSU-VSF08": {
        "name": "MadetoLastParts",
        "country": "US",
        "lead_time_days": 7,
        "on_time_delivery_pct": 95,
        "quality_rejection_pct": 0.7,
        "risk_level": "Low",
        "payment_terms": "Net 30",
        "spend_ytd_usd": 870_000,
        "recent_issues": "None reported.",
        "notes": "Second-best performing supplier. Strong quality record and responsive account team.",
    },
    "USSU-VSF09": {
        "name": "Trimax Electricals",
        "country": "IN",
        "lead_time_days": 45,
        "on_time_delivery_pct": 72,
        "quality_rejection_pct": 4.2,
        "risk_level": "High",
        "payment_terms": "Net 60",
        "spend_ytd_usd": 510_000,
        "recent_issues": "Factory capacity constraints Q1 2026. Customs delays on two shipments. Quality audit scheduled.",
        "notes": "High spend but high risk. Geographic concentration risk (India). Escalated to category manager.",
    },
    "USSU-VSF10": {
        "name": "Electronics Supply Inc.",
        "country": "US",
        "lead_time_days": 35,
        "on_time_delivery_pct": 82,
        "quality_rejection_pct": 1.9,
        "risk_level": "Medium",
        "payment_terms": "Net 45",
        "spend_ytd_usd": 430_000,
        "recent_issues": "Component shortage (semiconductor) caused 10-day delay in April 2026.",
        "notes": "Semiconductor supply dependency is the key risk. Recommended to qualify second source.",
    },
    "USSU_V8000": {
        "name": "EV Parts Inc.",
        "country": "US",
        "lead_time_days": 30,
        "on_time_delivery_pct": 84,
        "quality_rejection_pct": 1.6,
        "risk_level": "Medium",
        "payment_terms": "Net 45",
        "spend_ytd_usd": 620_000,
        "recent_issues": "New supplier (onboarded Q4 2025). Ramp-up phase ongoing.",
        "notes": "Strategically important for EV transition. Performance improving quarter-over-quarter.",
    },
    "USSU_V8001": {
        "name": "WaveCrest Labs",
        "country": "US",
        "lead_time_days": 14,
        "on_time_delivery_pct": 93,
        "quality_rejection_pct": 0.9,
        "risk_level": "Low",
        "payment_terms": "Net 30",
        "spend_ytd_usd": 740_000,
        "recent_issues": "None reported.",
        "notes": "Innovative supplier with strong R&D capability. Preferred partner for new product launches.",
    },
    "USSU_V8003": {
        "name": "Vendor V8003",
        "country": "MX",
        "lead_time_days": 60,
        "on_time_delivery_pct": 68,
        "quality_rejection_pct": 5.1,
        "risk_level": "High",
        "payment_terms": "Net 60",
        "spend_ytd_usd": 175_000,
        "recent_issues": "Three quality rejections in 2026. Shipment held at border in February. Corrective action plan requested.",
        "notes": "Lowest performing supplier. Sourcing team reviewing contract renewal. Cross-border logistics a recurring issue.",
    },
    "USSU_V8004": {
        "name": "Vendor V8004",
        "country": "MX",
        "lead_time_days": 28,
        "on_time_delivery_pct": 76,
        "quality_rejection_pct": 3.3,
        "risk_level": "High",
        "payment_terms": "Net 60",
        "spend_ytd_usd": 295_000,
        "recent_issues": "Packaging non-conformance flagged in Q1 2026. Two late deliveries.",
        "notes": "Geographic risk similar to V8003. Performance has not improved despite supplier development program.",
    },
    "USSU_V8005": {
        "name": "Vendor V8005",
        "country": "US",
        "lead_time_days": 14,
        "on_time_delivery_pct": 89,
        "quality_rejection_pct": 1.2,
        "risk_level": "Low",
        "payment_terms": "Net 30",
        "spend_ytd_usd": 390_000,
        "recent_issues": "None reported.",
        "notes": "Solid mid-tier domestic supplier. Reliable across all measured KPIs.",
    },
    "USSU_V8006": {
        "name": "Vendor V8006",
        "country": "CA",
        "lead_time_days": 21,
        "on_time_delivery_pct": 83,
        "quality_rejection_pct": 2.0,
        "risk_level": "Medium",
        "payment_terms": "Net 45",
        "spend_ytd_usd": 330_000,
        "recent_issues": "Currency fluctuation (CAD/USD) impacting pricing. One weather-related delay in January 2026.",
        "notes": "Canadian supplier with moderate FX exposure. Performance within acceptable range.",
    },
    "USSU_V8007": {
        "name": "Vendor V8007",
        "country": "MX",
        "lead_time_days": 45,
        "on_time_delivery_pct": 71,
        "quality_rejection_pct": 4.6,
        "risk_level": "High",
        "payment_terms": "Net 60",
        "spend_ytd_usd": 220_000,
        "recent_issues": "Quality complaints filed twice in 2026. Late delivery on three POs. Formal corrective action open.",
        "notes": "Under supplier performance improvement plan. Risk of contract termination if Q3 2026 targets not met.",
    },
}


def get_enrichment(supplier_id: str) -> dict:
    """Return metrics for a supplier ID, or an empty dict if not found."""
    return SUPPLIER_METRICS.get(supplier_id, {})


def enriched_supplier_docs() -> list:
    """
    Build a rich text document for each supplier in the enrichment registry.
    Used by sap_connector.fetch_suppliers() when the BP OData call is unavailable.
    """
    docs = []
    for supplier_id, m in SUPPLIER_METRICS.items():
        risk = m.get("risk_level", "Unknown")
        otd  = m.get("on_time_delivery_pct", "N/A")
        lt   = m.get("lead_time_days", "N/A")
        rej  = m.get("quality_rejection_pct", "N/A")
        docs.append(
            f"Supplier: {m.get('name', supplier_id)}\n"
            f"SAP Supplier ID: {supplier_id}\n"
            f"Country: {m.get('country', 'N/A')}\n"
            f"Lead Time: {lt} days\n"
            f"On-Time Delivery Rate: {otd}%\n"
            f"Quality Rejection Rate: {rej}%\n"
            f"Risk Level: {risk}\n"
            f"Payment Terms: {m.get('payment_terms', 'N/A')}\n"
            f"YTD Spend: ${m.get('spend_ytd_usd', 0):,}\n"
            f"Recent Issues: {m.get('recent_issues', 'None reported.')}\n"
            f"Notes: {m.get('notes', '')}"
        )
    return docs
