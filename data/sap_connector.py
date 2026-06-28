"""
SAP S/4HANA Cloud OData connector.

Supports two auth modes — whichever env vars are present wins:

  Mode 1 — SAP API Business Hub sandbox (no system access needed):
    SAP_API_HUB_KEY   API key from https://api.sap.com (profile → Show API Key)
    Base URL is always https://sandbox.api.sap.com

  Mode 2 — Direct S/4HANA tenant with Communication User (requires admin setup):
    SAP_BASE_URL      e.g. https://my413615.s4hana.cloud.sap
    SAP_USERNAME      Communication user created via Communication Arrangement
    SAP_PASSWORD      Password for that user
    SAP_CLIENT        SAP client number (default: 100)

Returns plain strings matching the format of erp_data.py / supplier_docs.py
so aria_graph.py needs no structural changes.
"""

import os
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()

# --- Auth mode detection ---
SAP_API_HUB_KEY = os.getenv("SAP_API_HUB_KEY", "")
SAP_BASE_URL    = os.getenv("SAP_BASE_URL", "").rstrip("/")
SAP_USERNAME    = os.getenv("SAP_USERNAME", "")
SAP_PASSWORD    = os.getenv("SAP_PASSWORD", "")
SAP_CLIENT      = os.getenv("SAP_CLIENT", "100")

if SAP_API_HUB_KEY:
    _BASE    = "https://sandbox.api.sap.com"
    _ODATA   = "/s4hanacloud/sap/opu/odata/sap"
    _MODE    = "hub"
else:
    _BASE    = SAP_BASE_URL
    _ODATA   = "/sap/opu/odata/sap"
    _MODE    = "direct"

_session = None

def _get_session():
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"Accept": "application/json"})
        if _MODE == "hub":
            _session.headers["APIKey"] = SAP_API_HUB_KEY
        else:
            _session.auth = HTTPBasicAuth(SAP_USERNAME, SAP_PASSWORD)
            _session.headers["sap-client"] = SAP_CLIENT
    return _session

def _get(service, entity, params=None):
    url = f"{_BASE}{_ODATA}/{service}/{entity}"
    resp = _get_session().get(url, params=params, timeout=20, verify=True)
    resp.raise_for_status()
    return resp.json().get("d", {}).get("results", [])

def _parse_sap_date(raw):
    """Convert /Date(milliseconds)/ to YYYY-MM-DD string."""
    if raw and raw.startswith("/Date("):
        try:
            ms = int(raw[6:-2])
            return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        except Exception:
            pass
    return raw or "N/A"


# ---------------------------------------------------------------------------
# Purchase Orders — API_PURCHASEORDER_PROCESS_SRV
# ---------------------------------------------------------------------------

def fetch_purchase_orders(top=100):
    rows = _get(
        "API_PURCHASEORDER_PROCESS_SRV",
        "A_PurchaseOrder",
        params={
            "$top": top,
            "$select": (
                "PurchaseOrder,Supplier,AddressName,PurchaseOrderType,"
                "DocumentCurrency,NetPaymentDays,PaymentTerms,"
                "CreationDate,PurchasingOrganization,PurchasingProcessingStatus"
            ),
            "$format": "json",
        },
    )

    docs = []
    for po in rows:
        status_code  = po.get("PurchasingProcessingStatus", "")
        status_label = {
            "01": "Created", "02": "In Process", "03": "Released",
            "04": "Completed", "05": "Fully Released", "06": "Closed"
        }.get(status_code, status_code or "Open")
        supplier_name = po.get("AddressName") or po.get("Supplier", "N/A")
        docs.append(
            f"ERP MODULE: Purchase Orders\n"
            f"PO Number: {po.get('PurchaseOrder', 'N/A')}\n"
            f"Supplier: {supplier_name}\n"
            f"Supplier ID: {po.get('Supplier', 'N/A')}\n"
            f"Type: {po.get('PurchaseOrderType', 'N/A')}\n"
            f"Currency: {po.get('DocumentCurrency', 'N/A')}\n"
            f"Payment Terms: {po.get('PaymentTerms', 'N/A')} (Net {po.get('NetPaymentDays', 'N/A')} days)\n"
            f"Created: {_parse_sap_date(po.get('CreationDate', ''))}\n"
            f"Purchasing Org: {po.get('PurchasingOrganization', 'N/A')}\n"
            f"ERP Status: {status_label}"
        )
    return docs


# ---------------------------------------------------------------------------
# Material / Inventory Stock — API_MATERIAL_STOCK_SRV
# ---------------------------------------------------------------------------

def fetch_material_stock(top=100):
    rows = _get(
        "API_MATERIAL_STOCK_SRV",
        "A_MatlStkInAcctMod",
        params={
            "$top": top,
            "$select": (
                "Material,Plant,StorageLocation,"
                "MatlWrhsStkQtyInMatlBaseUnit,MaterialBaseUnit"
            ),
            "$format": "json",
        },
    )

    docs = []
    for s in rows:
        qty = s.get("MatlWrhsStkQtyInMatlBaseUnit", "N/A")
        try:
            qty = float(qty)
        except (TypeError, ValueError):
            pass
        docs.append(
            f"ERP MODULE: Inventory Planning\n"
            f"Material: {s.get('Material', 'N/A')}\n"
            f"Plant: {s.get('Plant', 'N/A')}\n"
            f"Storage Location: {s.get('StorageLocation', 'N/A')}\n"
            f"Stock Quantity: {qty} {s.get('MaterialBaseUnit', '')}"
        )
    return docs


# ---------------------------------------------------------------------------
# Supplier Master Data — API_BUSINESS_PARTNER
# ---------------------------------------------------------------------------

def fetch_suppliers(top=100):
    # The Hub sandbox BP endpoint does not support filtering by Supplier field,
    # so we use the enrichment registry as the authoritative supplier source.
    # Enrichment is keyed by the real SAP Supplier IDs discovered from live PO data,
    # giving ARIA rich performance metrics (lead time, OTD, risk level, spend, etc.)
    # that the BP master data API does not carry.
    from data.supplier_enrichment import enriched_supplier_docs
    return enriched_supplier_docs()


# ---------------------------------------------------------------------------
# Connection test
# ---------------------------------------------------------------------------

def test_connection():
    """Probe each OData service with $top=1. Returns per-service status dict."""
    services = {
        "purchase_orders": ("API_PURCHASEORDER_PROCESS_SRV", "A_PurchaseOrder"),
        "material_stock":  ("API_MATERIAL_STOCK_SRV",        "A_MatlStkInAcctMod"),
        "suppliers":       ("API_BUSINESS_PARTNER",           "A_BusinessPartner"),
    }
    results = {}
    for key, (svc, entity) in services.items():
        try:
            rows = _get(svc, entity, params={"$top": 1, "$format": "json"})
            results[key] = {"ok": True, "rows_returned": len(rows)}
        except requests.HTTPError as e:
            results[key] = {"ok": False, "error": f"HTTP {e.response.status_code}: {e.response.text[:300]}"}
        except Exception as e:
            results[key] = {"ok": False, "error": str(e)}
    return results


# ---------------------------------------------------------------------------
# Convenience wrapper used by aria_graph.py
# ---------------------------------------------------------------------------

def fetch_all_sap_data():
    """Fetch all three sources; failed sources return [] so caller uses mock fallback."""
    out = {}
    for name, fn in [
        ("purchase_orders", fetch_purchase_orders),
        ("material_stock",  fetch_material_stock),
        ("suppliers",       fetch_suppliers),
    ]:
        try:
            out[name] = fn()
            print(f"[SAP] {name}: {len(out[name])} records fetched")
        except Exception as e:
            print(f"[SAP] {name}: failed ({e}) — will use mock data")
            out[name] = []
    return out


if __name__ == "__main__":
    print(f"SAP connector — mode: {_MODE}")
    if _MODE == "hub":
        print(f"Base URL : {_BASE}")
        print(f"API Key  : {SAP_API_HUB_KEY[:8]}...\n")
    else:
        print(f"Base URL : {_BASE}")
        print(f"Username : {SAP_USERNAME}")
        print(f"Client   : {SAP_CLIENT}\n")

    results = test_connection()
    for service, status in results.items():
        icon = "OK" if status["ok"] else "FAIL"
        print(f"[{icon}] {service}: {status}")
