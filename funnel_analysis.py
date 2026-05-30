"""
Marketing Funnel & Conversion Performance Analysis
Future Interns — Data Science & Analytics — Task 3 (2025)
Author: Elethu Mahlangeni
"""

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv("marketing_funnel_data.csv")
print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns\n")

# ─────────────────────────────────────────────────────────────────────────────
# 1. OVERALL FUNNEL METRICS
# ─────────────────────────────────────────────────────────────────────────────
overall = df[['Visitors','Leads','MQLs','SQLs','Customers','Revenue_USD','Marketing_Cost_USD']].sum()

print("=" * 55)
print("  OVERALL FUNNEL PERFORMANCE — FY 2025")
print("=" * 55)
print(f"  Visitors  : {overall['Visitors']:>10,.0f}")
print(f"  Leads     : {overall['Leads']:>10,.0f}  ({overall['Leads']/overall['Visitors']*100:.2f}% of visitors)")
print(f"  MQLs      : {overall['MQLs']:>10,.0f}  ({overall['MQLs']/overall['Leads']*100:.2f}% of leads)")
print(f"  SQLs      : {overall['SQLs']:>10,.0f}  ({overall['SQLs']/overall['MQLs']*100:.2f}% of MQLs)")
print(f"  Customers : {overall['Customers']:>10,.0f}  ({overall['Customers']/overall['SQLs']*100:.2f}% of SQLs)")
print(f"  End-to-end: {overall['Customers']/overall['Visitors']*100:.4f}%")
print(f"  Revenue   : ${overall['Revenue_USD']:>12,.2f}")
print(f"  Cost      : ${overall['Marketing_Cost_USD']:>12,.2f}")
print(f"  ROI       : {(overall['Revenue_USD']-overall['Marketing_Cost_USD'])/overall['Marketing_Cost_USD']*100:,.1f}%\n")

# ─────────────────────────────────────────────────────────────────────────────
# 2. CHANNEL-LEVEL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
ch = df.groupby('Channel')[['Visitors','Leads','MQLs','SQLs','Customers',
                              'Revenue_USD','Marketing_Cost_USD']].sum()
ch['V→Lead %']    = (ch['Leads']/ch['Visitors']*100).round(2)
ch['Lead→MQL %']  = (ch['MQLs']/ch['Leads']*100).round(2)
ch['MQL→SQL %']   = (ch['SQLs']/ch['MQLs']*100).round(2)
ch['SQL→Cust %']  = (ch['Customers']/ch['SQLs']*100).round(2)
ch['Conv %']      = (ch['Customers']/ch['Visitors']*100).round(4)
ch['ROI %']       = ((ch['Revenue_USD']-ch['Marketing_Cost_USD'])/ch['Marketing_Cost_USD']*100).round(1)
ch['Cost/Lead']   = (ch['Marketing_Cost_USD']/ch['Leads']).round(2)
ch['Cost/Cust']   = (ch['Marketing_Cost_USD']/ch['Customers']).round(2)

print("=" * 55)
print("  CHANNEL-LEVEL METRICS (sorted by ROI)")
print("=" * 55)
print(ch.sort_values('ROI %', ascending=False)[
    ['V→Lead %','Lead→MQL %','MQL→SQL %','SQL→Cust %','Conv %','ROI %','Cost/Lead','Cost/Cust']
].to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 3. MONTHLY TRENDS
# ─────────────────────────────────────────────────────────────────────────────
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
mo = df.groupby('Month_Num')[['Visitors','Leads','MQLs','SQLs','Customers','Revenue_USD']].sum().reset_index()
mo['Month'] = months
mo['Conv %'] = (mo['Customers']/mo['Visitors']*100).round(4)

print("\n" + "=" * 55)
print("  MONTHLY TREND SUMMARY")
print("=" * 55)
print(mo[['Month','Visitors','Leads','Customers','Revenue_USD','Conv %']].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 4. KEY INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  KEY INSIGHTS & DROP-OFF ANALYSIS")
print("=" * 55)

stages = {
    'Visitor→Lead':   overall['Leads']/overall['Visitors']*100,
    'Lead→MQL':       overall['MQLs']/overall['Leads']*100,
    'MQL→SQL':        overall['SQLs']/overall['MQLs']*100,
    'SQL→Customer':   overall['Customers']/overall['SQLs']*100,
}
for stage, rate in stages.items():
    drop = 100 - rate
    print(f"  {stage:<18}: {rate:5.1f}% pass  |  {drop:5.1f}% drop-off")

print(f"\n  Best channel by ROI    : {ch['ROI %'].idxmax()} ({ch['ROI %'].max():,.0f}%)")
print(f"  Best channel by conv   : {ch['Conv %'].idxmax()} ({ch['Conv %'].max():.4f}%)")
print(f"  Worst channel by ROI   : {ch['ROI %'].idxmin()} ({ch['ROI %'].min():.1f}%)")
print(f"  Cheapest lead source   : {ch['Cost/Lead'].idxmin()} (${ch['Cost/Lead'].min():.2f}/lead)")
print(f"  Most expensive CPC     : {ch['Cost/Cust'].idxmax()} (${ch['Cost/Cust'].max():.2f}/customer)")

print("\n  ✓ Analysis complete. See Marketing_Funnel_Dashboard.html for interactive visuals.")
