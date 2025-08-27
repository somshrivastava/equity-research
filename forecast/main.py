# main.py
import csv
from fundamentals import get_flow_metric, get_balance_metric
from momentum import get_momentum_returns

def main():
    YEARS_BACK = 10

    revenue    = get_flow_metric(
        ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
        "Revenue", years_back=YEARS_BACK
    )
    ebit       = get_flow_metric(["OperatingIncomeLoss"], "EBIT", years_back=YEARS_BACK)
    net_income = get_flow_metric(["NetIncomeLoss","ProfitLoss"], "NetIncome", years_back=YEARS_BACK)
    cogs       = get_flow_metric(["CostOfGoodsAndServicesSold","CostOfRevenue"], "COGS", years_back=YEARS_BACK)
    sga        = get_flow_metric(["SellingGeneralAndAdministrativeExpense"], "SGA", years_back=YEARS_BACK)

    cash        = get_balance_metric(["CashAndCashEquivalentsAtCarryingValue"], "Cash", years_back=YEARS_BACK)
    receivables = get_balance_metric(["AccountsReceivableNetCurrent"], "Receivables", years_back=YEARS_BACK)
    inventories = get_balance_metric(["InventoryNet"], "Inventories", years_back=YEARS_BACK)
    other_ca    = get_balance_metric(["OtherAssetsCurrent"], "OtherCurrentAssets", years_back=YEARS_BACK)
    ppe         = get_balance_metric(["PropertyPlantAndEquipmentNet"], "PPE", years_back=YEARS_BACK)
    other_a     = get_balance_metric(["OtherAssetsNoncurrent","OtherAssets"], "OtherAssets", years_back=YEARS_BACK)
    debt_cur    = get_balance_metric(["LongTermDebtCurrent","DebtCurrent"], "DebtCurrent", years_back=YEARS_BACK)
    ap          = get_balance_metric(["AccountsPayableCurrent"], "AccountsPayable", years_back=YEARS_BACK)
    liab_cur    = get_balance_metric(["LiabilitiesCurrent"], "LiabilitiesCurrent", years_back=YEARS_BACK)
    ocl         = get_balance_metric(["OtherLiabilitiesCurrent"], "OtherCurrentLiabilities", years_back=YEARS_BACK)
    tl          = get_balance_metric(["Liabilities"], "TotalLiabilities", years_back=YEARS_BACK)

    momentum_dict = get_momentum_returns("AAPL", years_back=YEARS_BACK)

    metrics = [revenue, cogs, ebit, sga, net_income,
               cash, receivables, inventories, other_ca, ppe,
               other_a, debt_cur, ap, liab_cur, ocl, tl]

    merged = {}
    for metric_list in metrics:
        for row in metric_list:
            period = row["Period"]  # e.g. "FY2020 Q1"
            if period not in merged:
                merged[period] = {"Period": period}
            merged[period].update(row)

    for period, row in merged.items():
        if "OtherCurrentLiabilities" not in row:
            if "LiabilitiesCurrent" in row:
                ap_val   = row.get("AccountsPayable", 0)
                debt_val = row.get("DebtCurrent", 0)
                liab_val = row.get("LiabilitiesCurrent", 0)
                row["OtherCurrentLiabilities"] = liab_val - ap_val - debt_val

    for period, row in merged.items():
        fy = period.split()[0]  # "FY2020"
        if fy in momentum_dict:
            row.update(momentum_dict[fy])
        else:
            for h in [1,3,6,9]:
                row[f"{h}m_return"] = ""

    def qsort(period: str):
        fy, q = period.split()
        return (int(fy.replace("FY","")), {"Q1":1,"Q2":2,"Q3":3,"Q4":4}[q])

    rows = sorted(merged.values(), key=lambda r: qsort(r["Period"]))

    fieldnames = [
        "Period",
        "Revenue","COGS","EBIT","SGA","NetIncome",
        "Cash","Receivables","Inventories","OtherCurrentAssets",
        "PPE","OtherAssets","DebtCurrent","AccountsPayable",
        "OtherCurrentLiabilities","LiabilitiesCurrent","TotalLiabilities",
        "1m_return","3m_return","6m_return","9m_return", "Price"
    ]

    # Fill missing cols with blanks
    for r in rows:
        for col in fieldnames:
            if col not in r:
                r[col] = ""

    with open("AAPL_fundamentals_last10y.csv","w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"✅ Saved AAPL_fundamentals_last10y.csv with {len(rows)} rows.")

if __name__ == "__main__":
    main()
