import pandas as pd
import numpy as np

def compute_value_creation(growth_df, capital_df):
    # Allow passing CapitalEfficiencyResult (dataclass) or a DataFrame
    if hasattr(capital_df, "df"):
        capital_df = capital_df.df
    """
    growth_df  -> output of annual_normalized.py
    capital_df -> output of capital_efficiency.py
    """

    df = pd.merge(
        growth_df,
        capital_df,
        on="Ticker",
        how="left"
    )

    # Choose best growth metric available
    def pick_growth(row):
        for col in ["Revenue_CAGR_%", "EBITDA_CAGR_%", "NetIncome_CAGR_%"]:
            if pd.notna(row.get(col)):
                return row[col]
        return None

    df["Growth_%"] = df.apply(pick_growth, axis=1)

    # Value creation score
    df["Value_Creation_Score"] = (
        df["ROIC_%"] / 100 * df["Growth_%"] / 100
    )

    # Quadrant classification
    def classify(row):
        roic = row.get("ROIC_%")
        g = row.get("Growth_%")
        if pd.isna(roic) or pd.isna(g):
            return "Insufficient data"

        if roic >= 10 and g >= 5:
            return "Compounder"
        if roic >= 10 and g < 5:
            return "Cash Cow"
        if roic < 10 and g >= 5:
            return "Value Destroyer"
        return "Melting Ice Cube"

    df["Economic_Quadrant"] = df.apply(classify, axis=1)

    # Investment conclusion
    def conclude(row):
        q = row["Economic_Quadrant"]
        if q == "Compounder":
            return "Premium justified"
        if q == "Cash Cow":
            return "Yield / buybacks play"
        if q == "Value Destroyer":
            return "Avoid / value trap"
        if q == "Melting Ice Cube":
            return "Fully priced / avoid"
        if q == "Insufficient data":
            return "Incomplete data"
        return "Incomplete data"

    df["Investment_Conclusion"] = df.apply(conclude, axis=1)

    return df[[
        "Ticker",
        "ROIC_%",
        "Growth_%",
        "Value_Creation_Score",
        "Economic_Quadrant",
        "Investment_Conclusion"
    ]]
