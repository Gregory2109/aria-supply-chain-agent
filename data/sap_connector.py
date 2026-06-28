"""
SAP S/4HANA Cloud OData connector — fetches live purchase orders, material
stock, and supplier master data via three Communication Arrangement APIs.

Required env vars:
  SAP_BASE_URL   e.g. https://my413615.s4hana.cloud.sap
  SAP_USERNAME   Communication user created in the SAP Fiori launchpad
  SAP_PASSWORD   Password for that communication user
  SAP_CLIENT     SAP client number (default: 100)

Returns lists of plain strings formatted to match the mock data in
erp_data.py / supplier_docs.py so aria_graph.py needs no structural changes.
"""

import os
import requests
from requests.auth import HTTPBasicAuth

SAP_BASE_URL = os.getenv("SAP_BASE_URL", "").rstrip("/")
SAP_USERNAME = os.getenv("SAP_USERNAME", "")
SAP_PASSWORD = os.getenv("SAP_PASSWORD", "")
SAP_CLIENT   = os.getenv("SAP_CLIENT", "100")

ODATA_BASE = "/sap/opu/odata/sap"

_session = None

def _get_session():
    global _session
    if _session is None:
        _session = requests.Session()
        _session.auth = HTTPBasicAuth(SAP_USERNAME, SAP_PASSWORD)
        _session.headers.update({
            "Accept": "application/json",
            "sap-client": SAP_CLIENT,
        })
    return _session

def _get(service, entity, params=None):
    url = f"{SAP_BASE_URL}{ODATA_BASE}/{service}/{entity}"
    resp = _get_session().get(url, params=params, timeout=20, verify=True)
    resp.raise_for_status()
    return resp.json().get("d", {}).get("results", [])


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
                "PurchaseOrder,Supplier,PurchaseOrderType,"
                "DocumentCurrency,NetPaymentDays,TotalNetAmount,"
                "CreationDate,PurchasingOrganization,PurchaseOrderStatus"
            ),
            "$format": "json",
        },
    )

    docs = []
    for po in rows:
        status_code = po.get("PurchaseOrderStatus", "")
        status_label = {"": "Draft", "B": "In Process", "N": "Complete"}.get(status_code, status_code)
        created = po.get("CreationDate", "")
        # SAP returns dates as /Date(milliseconds)/ — convert to readable string
        if created.startswith("/Date("):
            try:
                ms = int(created[6:-2])
                from datetime import datetime, timezone
                created = datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            except Exception:
                pass

        docs.append(
            f"ERP MODULE: Purchase Orders\n"
            f"PO Number: {po.get('PurchaseOrder', 'N/A')}\n"
            f"Supplier ID: {po.get('Supplier', 'N/A')}\n"
            f"Type: {po.get('PurchaseOrderType', 'N/A')}\n"
            f"Total Amount: {po.get('TotalNetAmount', 'N/A')} {po.get('DocumentCurrency', '')}\n"
            f"Payment Terms: Net {po.get('NetPaymentDays', 'N/A')} days\n"
            f"Created: {created}\n"
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
    rows = _get(
        "API_BUSINESS_PARTNER",
        "A_BusinessPartner",
        params={
            "$top": top,
            "$select": (
                "BusinessPartner,BusinessPartnerFullName,"
                "BusinessPartnerGrouping,Country,Language"
            ),
            "$filter": "BusinessPartnerGrouping eq 'KRED'",  # KRED = vendor/supplier
            "$format": "json",
        },
    )

    # Fallback: if the KRED filter returns nothing (sandbox may use different
    # groupings), re-fetch without the filter so we get whatever exists.
    if not rows:
        rows = _get(
            "API_BUSINESS_PARTNER",
            "A_BusinessPartner",
            params={
                "$top": top,
                "$select": (
                    "BusinessPartner,BusinessPartnerFullName,"
                    "BusinessPartnerGrouping,Country,Language"
                ),
                "$format": "json",
            },
        )

    docs = []
    for bp in rows:
        name = bp.get("BusinessPartnerFullName") or bp.get("BusinessPartner", "N/A")
        docs.append(
            f"Supplier: {name}\n"
            f"SAP Business Partner ID: {bp.get('BusinessPartner', 'N/A')}\n"
            f"Grouping: {bp.get('BusinessPartnerGrouping', 'N/A')}\n"
            f"Country: {bp.get('Country', 'N/A')}"
        )
    return docs


# ---------------------------------------------------------------------------
# Connection test — call this to verify credentials before reindexing
# ---------------------------------------------------------------------------

def test_connection():
    """
    Tries each of the three OData services with a $top=1 probe.
    Returns a dict with per-service status so callers can surface errors clearly.
    """
    services = {
        "purchase_orders": ("API_PURCHASEORDER_PROCESS_SRV", "A_PurchaseOrder"),
        "material_stock":  ("API_MATERIAL_STOCK_SRV",         "A_MatlStkInAcctMod"),
        "suppliers":       ("API_BUSINESS_PARTNER",            "A_BusinessPartner"),
    }
    results = {}
    for key, (svc, entity) in services.items():
        try:
            rows = _get(svc, entity, params={"$top": 1, "$format": "json"})
            results[key] = {"ok": True, "rows_returned": len(rows)}
        except requests.HTTPError as e:
            results[key] = {"ok": False, "error": f"HTTP {e.response.status_code}: {e.response.text[:200]}"}
        except Exception as e:
            results[key] = {"ok": False, "error": str(e)}
    return results


# ---------------------------------------------------------------------------
# Convenience wrapper used by aria_graph.py
# ---------------------------------------------------------------------------

def fetch_all_sap_data():
    """
    Fetches all three data sources. Each failed source logs a warning and
    returns an empty list — the caller falls back to mock data per source.
    """
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
    print("Testing SAP connection...")
    print(f"Base URL : {SAP_BASE_URL}")
    print(f"Username : {SAP_USERNAME}")
    print(f"Client   : {SAP_CLIENT}\n")

    results = test_connection()
    for service, status in results.items():
        icon = "OK" if status["ok"] else "FAIL"
        print(f"[{icon}] {service}: {status}")
