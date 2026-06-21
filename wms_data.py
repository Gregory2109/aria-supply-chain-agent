# Simulates data pulled from a WMS (Warehouse Management System)
# In production this would be a live API call to Manhattan, SAP WM, etc.

wms_data = [
    """WMS MODULE: Inbound Shipments
       Supplier: GlobalMetal Co
       Shipment ID: SHP-2026-0892
       Expected Arrival: 3 days ago (OVERDUE)
       Quantity: 500 units
       Carrier: China Freight Express
       Last GPS Update: Shanghai port — customs hold
       WMS Status: DELAYED — customs clearance pending
       Action Required: Notify procurement team immediately""",

    """WMS MODULE: Inbound Shipments
       Supplier: QuickShip Ltd
       Shipment ID: SHP-2026-1023
       Expected Arrival: Tomorrow
       Quantity: 200 units
       Carrier: FedEx Express
       Last GPS Update: Chicago distribution center
       WMS Status: ON TRACK — no issues""",

    """WMS MODULE: Inbound Shipments
       Supplier: AsiaComponents
       Shipment ID: SHP-2026-0967
       Expected Arrival: 18 days
       Quantity: 350 units
       Carrier: Vietnam Logistics Co
       Last GPS Update: Ho Chi Minh City warehouse
       WMS Status: PROCESSING — not yet shipped
       Risk: Supplier confirmed capacity constraints may delay dispatch""",

    """WMS MODULE: Warehouse Inventory
       Location: Chicago DC
       Total SKUs tracked: 1,247
       Low stock alerts: 3 (AsiaComponents x2, GlobalMetal Co x1)
       Overdue inbound shipments: 2
       Average receiving time: 2.3 days
       WMS Status: OPERATIONAL — 2 critical alerts active""",

    """WMS MODULE: Returns & Quality
       Supplier: GlobalMetal Co
       Returns this quarter: 47 units ($8,400 value)
       Return reason: Dimensional non-conformance
       Quality hold: 120 units pending inspection
       WMS Status: QUALITY HOLD ACTIVE
       Action Required: QA team inspection scheduled for tomorrow"""
]