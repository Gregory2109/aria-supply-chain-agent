# Simulates data pulled from an ERP system (e.g. SAP, Oracle)
# In production this would be a live API call to SAP or Oracle ERP

erp_data = [
    """ERP MODULE: Purchase Orders
       Supplier: GlobalMetal Co
       Open POs: 3
       Total Value: $142,000
       Oldest Open PO: 67 days (overdue by 22 days)
       Payment Terms: Net 30
       Last Invoice: $48,000 — disputed due to quality defects
       ERP Status: FLAGGED — escalation required""",

    """ERP MODULE: Purchase Orders
       Supplier: QuickShip Ltd
       Open POs: 7
       Total Value: $89,500
       Oldest Open PO: 4 days
       Payment Terms: Net 45
       Last Invoice: $12,300 — cleared
       ERP Status: HEALTHY""",

    """ERP MODULE: Inventory Planning
       Supplier: AsiaComponents
       Current Stock: 142 units (12 days of supply remaining)
       Reorder Point: 200 units
       Safety Stock: 50 units
       Status: BELOW REORDER POINT — urgent replenishment needed
       Next Scheduled Delivery: 18 days away
       Risk: Stockout likely in 12 days if delivery delayed""",

    """ERP MODULE: Inventory Planning
       Supplier: FastParts Inc
       Current Stock: 890 units (45 days of supply remaining)
       Reorder Point: 300 units
       Safety Stock: 100 units
       Status: HEALTHY — adequate stock levels
       Next Scheduled Delivery: 14 days away""",

    """ERP MODULE: Inventory Planning
       Supplier: MedParts GmbH
       Current Stock: 320 units (21 days of supply remaining)
       Reorder Point: 250 units
       Safety Stock: 75 units
       Status: MONITOR — approaching reorder point
       Note: Supplier confirmed 15% price increase effective Q3 2026"""
]