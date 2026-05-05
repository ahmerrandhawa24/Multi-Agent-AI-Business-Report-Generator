import pandas as pd
import json
import os
from datetime import datetime

def run_daily_etl():
    print(f"ETL started: {datetime.now()}")

    # Extract
    df = pd.read_csv("data/sales_data.csv")

    # Transform
    df["Avg_Order_Value"] = (df["Revenue_PKR"] / df["Orders"]).round(2)
    df["Return_Rate_Pct"] = (df["Returns"] / df["Orders"] * 100).round(2)
    df = df.drop_duplicates()

    # Load
    df.to_csv("data/sales_clean.csv", index=False)

    # Log
    log = {
        "timestamp"     : str(datetime.now()),
        "rows_processed": len(df),
        "status"        : "success"
    }
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/etl_log_latest.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"ETL complete: {len(df)} rows processed")

if __name__ == "__main__":
    run_daily_etl()
