# app.py
# Required packages:
# streamlit
# pandas
# numpy
# plotly
# openpyxl

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="SKU-Level Retail Supply Chain Performance Dashboard",
    layout="wide"
)


# ============================================================
# Configuration
# ============================================================

APP_TITLE = "SKU-Level Retail Supply Chain Performance Dashboard"
APP_SUBTITLE = (
    "A descriptive analytics tool for evaluating SKU revenue, inventory, "
    "logistics, supplier, and quality performance."
)

POSSIBLE_FILES = [
    "Supply_Chain_Data.xlsx",
    "Supply_Chain_Data.csv"
]

REQUIRED_COLUMNS = [
    "sku",
    "product_type",
    "revenue_generated",
    "stock_levels",
    "number_of_products_sold",
    "lead_time",
    "shipping_costs",
    "supplier_name",
    "defect_rates"
]

NUMERIC_COLUMNS = [
    "revenue_generated",
    "price",
    "stock_levels",
    "lead_time",
    "lead_times",
    "shipping_costs",
    "manufacturing_costs",
    "defect_rates",
    "order_quantities",
    "availability",
    "number_of_products_sold",
    "production_volumes",
    "manufacturing_lead_time",
    "costs"
]

FILTER_COLUMNS = {
    "Product type": "product_type",
    "Supplier name": "supplier_name",
    "Shipping carrier": "shipping_carriers",
    "Transportation mode": "transportation_modes",
    "Route": "routes",
    "Location": "location",
    "Inspection result": "inspection_results",
    "Defect risk category": "defect_risk_category",
    "Inventory risk category": "inventory_risk_category",
    "Lead time risk category": "lead_time_risk_category"
}


# ============================================================
# Utility Functions
# ============================================================

def normalize_column_name(column: str) -> str:
    """
    Normalize column names by stripping whitespace, converting to lowercase,
    and replacing spaces with underscores.
    """
    return (
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def export_csv(df: pd.DataFrame) -> bytes:
    """
    Convert a dataframe to CSV bytes for Streamlit download buttons.
    """
    return df.to_csv(index=False).encode("utf-8")


def find_dataset_file() -> Path | None:
    """
    Search for the expected dataset file in the same directory as app.py.
    The app supports either Excel or CSV input.
    """
    current_dir = Path(__file__).parent if "__file__" in globals() else Path.cwd()

    for file_name in POSSIBLE_FILES:
        file_path = current_dir / file_name
        if file_path.exists():
            return file_path

    return None


@st.cache_data(show_spinner=False)
def load_data() -> Tuple[pd.DataFrame | None, str | None]:
    """
    Load the dataset from the same folder as app.py.
    Excel and CSV formats are supported.
    """
    dataset_path = find_dataset_file()

    if dataset_path is None:
        return None, (
            "Dataset file not found. Please place either "
            "'Supply_Chain_Data.xlsx' or 'Supply_Chain_Data.csv' "
            "in the same folder as app.py."
        )

    try:
        if dataset_path.suffix.lower() == ".xlsx":
            df = pd.read_excel(dataset_path, engine="openpyxl")
        elif dataset_path.suffix.lower() == ".csv":
            df = pd.read_csv(dataset_path)
        else:
            return None, "Unsupported file format. Please use .xlsx or .csv."

        if df.empty:
            return None, "The dataset loaded successfully, but it is empty."

        return df, None

    except Exception as error:
        return None, f"Unable to load dataset safely. Error: {error}"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize columns, consolidate aliases, remove duplicate rows,
    and convert relevant columns to numeric safely.
    """
    df = df.copy()

    df.columns = [normalize_column_name(col) for col in df.columns]

    # Consolidate lead_time and lead_times aliases into one consistent field.
    if "lead_time" not in df.columns and "lead_times" in df.columns:
        df["lead_time"] = df["lead_times"]

    if "lead_time" in df.columns and "lead_times" in df.columns:
        df["lead_time"] = df["lead_time"].combine_first(df["lead_times"])
        df = df.drop(columns=["lead_times"])

    # Remove exact duplicate rows.
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        st.warning(f"{duplicate_count:,} duplicate row(s) were found and removed.")
        df = df.drop_duplicates()

    # Convert relevant numeric columns safely.
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def validate_columns(df: pd.DataFrame) -> bool:
    """
    Validate required fields before analysis.
    If fields are missing, stop downstream processing and show a clear message.
    """
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_columns:
        st.error(
            "The dataset is missing required column(s): "
            + ", ".join(missing_columns)
        )

        st.write("Available columns in the dataset:")
        st.write(list(df.columns))

        return False

    return True


def create_risk_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create rule-based inventory, lead time, quality, and cost burden risk flags.
    """
    df = df.copy()

    sales_col = "number_of_products_sold"
    stock_col = "stock_levels"
    lead_col = "lead_time"
    defect_col = "defect_rates"
    shipping_col = "shipping_costs"
    manufacturing_col = "manufacturing_costs"

    stock_75 = df[stock_col].quantile(0.75) if stock_col in df.columns else np.nan
    stock_25 = df[stock_col].quantile(0.25) if stock_col in df.columns else np.nan

    sales_75 = df[sales_col].quantile(0.75) if sales_col in df.columns else np.nan
    sales_25 = df[sales_col].quantile(0.25) if sales_col in df.columns else np.nan

    lead_avg = df[lead_col].mean() if lead_col in df.columns else np.nan
    lead_75 = df[lead_col].quantile(0.75) if lead_col in df.columns else np.nan

    defect_75 = df[defect_col].quantile(0.75) if defect_col in df.columns else np.nan

    shipping_75 = (
        df[shipping_col].quantile(0.75)
        if shipping_col in df.columns
        else np.nan
    )

    manufacturing_75 = (
        df[manufacturing_col].quantile(0.75)
        if manufacturing_col in df.columns
        else np.nan
    )

    # Inventory risk: high stock with low sales may indicate overstock.
    df["overstock_risk_flag"] = (
        (df[stock_col] >= stock_75)
        & (df[sales_col] <= sales_25)
    )

    # Inventory risk: low stock with high sales may indicate stockout exposure.
    df["stockout_risk_flag"] = (
        (df[stock_col] <= stock_25)
        & (df[sales_col] >= sales_75)
    )

    # Lead time risk: above average or above 75th percentile indicates slower replenishment.
    df["long_lead_time_risk_flag"] = (
        (df[lead_col] > lead_avg)
        | (df[lead_col] >= lead_75)
    )

    # Quality risk: high defect rate relative to the dataset.
    df["high_defect_risk_flag"] = df[defect_col] >= defect_75

    # Cost burden risk: shipping cost above 75th percentile.
    if shipping_col in df.columns:
        df["shipping_cost_burden_risk_flag"] = df[shipping_col] >= shipping_75
    else:
        df["shipping_cost_burden_risk_flag"] = False

    # Cost burden risk: manufacturing cost above 75th percentile.
    if manufacturing_col in df.columns:
        df["manufacturing_cost_burden_risk_flag"] = (
            df[manufacturing_col] >= manufacturing_75
        )
    else:
        df["manufacturing_cost_burden_risk_flag"] = False

    risk_flag_columns = [
        "overstock_risk_flag",
        "stockout_risk_flag",
        "long_lead_time_risk_flag",
        "high_defect_risk_flag",
        "shipping_cost_burden_risk_flag",
        "manufacturing_cost_burden_risk_flag"
    ]

    df["risk_flag_count"] = df[risk_flag_columns].sum(axis=1)

    def build_risk_reasons(row: pd.Series) -> str:
        reasons = []

        if row.get("overstock_risk_flag", False):
            reasons.append("High stock and low sales")
        if row.get("stockout_risk_flag", False):
            reasons.append("Low stock and high sales")
        if row.get("long_lead_time_risk_flag", False):
            reasons.append("Long supplier lead time")
        if row.get("high_defect_risk_flag", False):
            reasons.append("High defect rate")
        if row.get("shipping_cost_burden_risk_flag", False):
            reasons.append("High shipping cost")
        if row.get("manufacturing_cost_burden_risk_flag", False):
            reasons.append("High manufacturing cost")

        return "; ".join(reasons) if reasons else "No major risk flag"

    df["risk_reasons"] = df.apply(build_risk_reasons, axis=1)

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create analytical features used in the dashboard.
    Calculations avoid fabricating missing values and use NaN where needed.
    """
    df = df.copy()

    # Revenue per unit sold measures how much revenue is generated for each sold unit.
    df["revenue_per_unit_sold"] = np.where(
        df["number_of_products_sold"] > 0,
        df["revenue_generated"] / df["number_of_products_sold"],
        np.nan
    )

    # Stock-to-sales ratio compares current inventory to sales volume.
    df["stock_to_sales_ratio"] = np.where(
        df["number_of_products_sold"] > 0,
        df["stock_levels"] / df["number_of_products_sold"],
        np.nan
    )

    # Shipping cost per unit estimates logistics burden per ordered or sold unit.
    if "order_quantities" in df.columns:
        df["shipping_cost_per_unit"] = np.where(
            df["order_quantities"] > 0,
            df["shipping_costs"] / df["order_quantities"],
            np.nan
        )
    else:
        df["shipping_cost_per_unit"] = np.where(
            df["number_of_products_sold"] > 0,
            df["shipping_costs"] / df["number_of_products_sold"],
            np.nan
        )

    # Manufacturing cost per unit estimates production cost burden per production unit.
    if "manufacturing_costs" in df.columns and "production_volumes" in df.columns:
        df["manufacturing_cost_per_unit"] = np.where(
            df["production_volumes"] > 0,
            df["manufacturing_costs"] / df["production_volumes"],
            np.nan
        )
    elif "manufacturing_costs" in df.columns:
        df["manufacturing_cost_per_unit"] = np.where(
            df["number_of_products_sold"] > 0,
            df["manufacturing_costs"] / df["number_of_products_sold"],
            np.nan
        )
    else:
        df["manufacturing_cost_per_unit"] = np.nan

    df = create_risk_flags(df)

    defect_75 = df["defect_rates"].quantile(0.75)
    defect_50 = df["defect_rates"].quantile(0.50)

    def defect_category(value: float) -> str:
        if pd.isna(value):
            return "Unknown"
        if value >= defect_75:
            return "High Defect Risk"
        if value >= defect_50:
            return "Moderate Defect Risk"
        return "Low Defect Risk"

    df["defect_risk_category"] = df["defect_rates"].apply(defect_category)

    def inventory_category(row: pd.Series) -> str:
        if row.get("overstock_risk_flag", False):
            return "Overstock Risk"
        if row.get("stockout_risk_flag", False):
            return "Stockout Risk"
        return "Low Inventory Risk"

    df["inventory_risk_category"] = df.apply(inventory_category, axis=1)

    def lead_time_category(row: pd.Series) -> str:
        if row.get("long_lead_time_risk_flag", False):
            return "Long Lead Time Risk"
        return "Low Lead Time Risk"

    df["lead_time_risk_category"] = df.apply(lead_time_category, axis=1)

    def overall_risk_category(flag_count: int) -> str:
        if flag_count >= 3:
            return "High Risk"
        if flag_count >= 1:
            return "Moderate Risk"
        return "Low Risk"

    df["overall_operational_risk"] = df["risk_flag_count"].apply(overall_risk_category)

    return df


def compute_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute dashboard KPI values.
    """
    total_revenue = df["revenue_generated"].sum(skipna=True)
    total_skus = df["sku"].nunique()
    avg_revenue_per_sku = total_revenue / total_skus if total_skus else np.nan

    high_risk_count = (
        df[df["overall_operational_risk"] == "High Risk"]["sku"].nunique()
        if "overall_operational_risk" in df.columns
        else 0
    )

    return {
        "total_revenue": total_revenue,
        "total_skus": total_skus,
        "avg_revenue_per_sku": avg_revenue_per_sku,
        "avg_stock_level": df["stock_levels"].mean(skipna=True),
        "avg_lead_time": df["lead_time"].mean(skipna=True),
        "avg_shipping_cost": df["shipping_costs"].mean(skipna=True),
        "avg_defect_rate": df["defect_rates"].mean(skipna=True),
        "high_risk_skus": high_risk_count
    }


def format_currency(value: float) -> str:
    """
    Format numeric values as currency.
    """
    if pd.isna(value):
        return "N/A"
    return f"${value:,.2f}"


def format_number(value: float) -> str:
    """
    Format numeric values as general numbers.
    """
    if pd.isna(value):
        return "N/A"
    return f"{value:,.2f}"


def format_integer(value: float) -> str:
    """
    Format numeric values as integer-like values.
    """
    if pd.isna(value):
        return "N/A"
    return f"{int(value):,}"


def add_sidebar_filter(
    df: pd.DataFrame,
    label: str,
    column: str
) -> List[str]:
    """
    Build a dynamic sidebar multiselect filter with an All option.
    """
    if column not in df.columns:
        return ["All"]

    options = (
        df[column]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

    choices = ["All"] + options

    return st.sidebar.multiselect(
        label,
        options=choices,
        default=["All"]
    )


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all selected sidebar filters to the dataset.
    """
    filtered_df = df.copy()

    st.sidebar.header("Dashboard Filters")

    selected_filters = {}

    for label, column in FILTER_COLUMNS.items():
        selected_filters[column] = add_sidebar_filter(df, label, column)

    for column, selected_values in selected_filters.items():
        if column not in filtered_df.columns:
            continue

        if "All" not in selected_values and selected_values:
            filtered_df = filtered_df[
                filtered_df[column].astype(str).isin(selected_values)
            ]

    return filtered_df


def create_supplier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create supplier-level summary table.
    """
    summary = (
        df.groupby("supplier_name", dropna=False)
        .agg(
            total_skus=("sku", "nunique"),
            total_revenue=("revenue_generated", "sum"),
            avg_lead_time=("lead_time", "mean"),
            avg_defect_rate=("defect_rates", "mean"),
            avg_shipping_cost=("shipping_costs", "mean"),
            high_risk_skus=(
                "overall_operational_risk",
                lambda x: (x == "High Risk").sum()
            )
        )
        .reset_index()
        .sort_values("total_revenue", ascending=False)
    )

    return summary


def build_charts(df: pd.DataFrame) -> None:
    """
    Build required Plotly charts in a clean two-column dashboard layout.
    """
    st.subheader("Performance Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        revenue_by_product = (
            df.groupby("product_type", dropna=False)["revenue_generated"]
            .sum()
            .reset_index()
            .sort_values("revenue_generated", ascending=False)
        )

        fig = px.bar(
            revenue_by_product,
            x="product_type",
            y="revenue_generated",
            title="Revenue by Product Type",
            labels={
                "product_type": "Product Type",
                "revenue_generated": "Revenue Generated"
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        top_skus = (
            df.groupby("sku", dropna=False)["revenue_generated"]
            .sum()
            .reset_index()
            .sort_values("revenue_generated", ascending=False)
            .head(10)
        )

        fig = px.bar(
            top_skus,
            x="sku",
            y="revenue_generated",
            title="Top 10 SKUs by Revenue",
            labels={
                "sku": "SKU",
                "revenue_generated": "Revenue Generated"
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig = px.scatter(
            df,
            x="number_of_products_sold",
            y="stock_levels",
            color="inventory_risk_category",
            hover_data=[
                "sku",
                "product_type",
                "supplier_name",
                "stock_levels",
                "number_of_products_sold"
            ],
            title="Stock Levels vs. Number of Products Sold",
            labels={
                "number_of_products_sold": "Number of Products Sold",
                "stock_levels": "Stock Levels",
                "inventory_risk_category": "Inventory Risk"
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        supplier_lead_time = (
            df.groupby("supplier_name", dropna=False)["lead_time"]
            .mean()
            .reset_index()
            .sort_values("lead_time", ascending=False)
        )

        fig = px.bar(
            supplier_lead_time,
            x="supplier_name",
            y="lead_time",
            title="Lead Time by Supplier",
            labels={
                "supplier_name": "Supplier Name",
                "lead_time": "Average Lead Time"
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    col5, col6 = st.columns(2)

    with col5:
        if "shipping_carriers" in df.columns:
            cost_group_col = "shipping_carriers"
            chart_title = "Shipping Cost by Carrier"
            x_label = "Shipping Carrier"
        elif "transportation_modes" in df.columns:
            cost_group_col = "transportation_modes"
            chart_title = "Shipping Cost by Transportation Mode"
            x_label = "Transportation Mode"
        else:
            cost_group_col = None

        if cost_group_col:
            shipping_summary = (
                df.groupby(cost_group_col, dropna=False)["shipping_costs"]
                .mean()
                .reset_index()
                .sort_values("shipping_costs", ascending=False)
            )

            fig = px.bar(
                shipping_summary,
                x=cost_group_col,
                y="shipping_costs",
                title=chart_title,
                labels={
                    cost_group_col: x_label,
                    "shipping_costs": "Average Shipping Cost"
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                "Shipping carrier or transportation mode columns are unavailable."
            )

    with col6:
        fig = px.histogram(
            df,
            x="defect_rates",
            nbins=20,
            title="Defect Rate Distribution",
            labels={
                "defect_rates": "Defect Rate"
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    col7, col8 = st.columns(2)

    with col7:
        fig = px.scatter(
            df,
            x="defect_rates",
            y="revenue_generated",
            color="overall_operational_risk",
            hover_data=[
                "sku",
                "product_type",
                "supplier_name",
                "risk_reasons"
            ],
            title="Revenue vs. Defect Rate",
            labels={
                "defect_rates": "Defect Rate",
                "revenue_generated": "Revenue Generated",
                "overall_operational_risk": "Operational Risk"
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    with col8:
        risk_summary = (
            df["overall_operational_risk"]
            .value_counts()
            .reset_index()
        )
        risk_summary.columns = ["overall_operational_risk", "sku_count"]

        fig = px.bar(
            risk_summary,
            x="overall_operational_risk",
            y="sku_count",
            title="Operational Risk Table",
            labels={
                "overall_operational_risk": "Overall Operational Risk",
                "sku_count": "SKU Count"
            }
        )
        st.plotly_chart(fig, use_container_width=True)


def display_kpis(df: pd.DataFrame) -> None:
    """
    Display KPI cards at the top of the dashboard.
    """
    kpis = compute_kpis(df)

    st.subheader("Executive KPI Summary")

    row1 = st.columns(4)

    row1[0].metric("Total Revenue", format_currency(kpis["total_revenue"]))
    row1[1].metric("Total SKUs", format_integer(kpis["total_skus"]))
    row1[2].metric(
        "Average Revenue per SKU",
        format_currency(kpis["avg_revenue_per_sku"])
    )
    row1[3].metric(
        "Average Stock Level",
        format_number(kpis["avg_stock_level"])
    )

    row2 = st.columns(4)

    row2[0].metric("Average Lead Time", format_number(kpis["avg_lead_time"]))
    row2[1].metric(
        "Average Shipping Cost",
        format_currency(kpis["avg_shipping_cost"])
    )
    row2[2].metric(
        "Average Defect Rate",
        format_number(kpis["avg_defect_rate"])
    )
    row2[3].metric("High-Risk SKUs", format_integer(kpis["high_risk_skus"]))


def display_tables_and_downloads(df: pd.DataFrame) -> None:
    """
    Display required dashboard tables and CSV download buttons.
    """
    st.subheader("SKU-Level Tables and Downloads")

    top_skus_table = (
        df.sort_values("revenue_generated", ascending=False)
        .head(10)
        .copy()
    )

    high_risk_table = (
        df[df["overall_operational_risk"] == "High Risk"]
        .sort_values("risk_flag_count", ascending=False)
        .copy()
    )

    supplier_summary = create_supplier_summary(df)

    table_tabs = st.tabs([
        "Filtered SKU-Level Data",
        "Top SKUs by Revenue",
        "High-Risk SKUs",
        "Supplier Performance Summary"
    ])

    with table_tabs[0]:
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            label="Download Filtered Dataset CSV",
            data=export_csv(df),
            file_name="filtered_sku_supply_chain_data.csv",
            mime="text/csv"
        )

    with table_tabs[1]:
        st.dataframe(top_skus_table, use_container_width=True, hide_index=True)

    with table_tabs[2]:
        high_risk_columns = [
            col for col in [
                "sku",
                "product_type",
                "supplier_name",
                "revenue_generated",
                "stock_levels",
                "number_of_products_sold",
                "lead_time",
                "defect_rates",
                "shipping_costs",
                "manufacturing_costs",
                "risk_flag_count",
                "overall_operational_risk",
                "risk_reasons"
            ]
            if col in high_risk_table.columns
        ]

        st.dataframe(
            high_risk_table[high_risk_columns],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            label="Download High-Risk SKU Table CSV",
            data=export_csv(high_risk_table),
            file_name="high_risk_sku_table.csv",
            mime="text/csv"
        )

    with table_tabs[3]:
        st.dataframe(supplier_summary, use_container_width=True, hide_index=True)

        st.download_button(
            label="Download Supplier Summary CSV",
            data=export_csv(supplier_summary),
            file_name="supplier_performance_summary.csv",
            mime="text/csv"
        )


def display_data_quality_notes(df: pd.DataFrame) -> None:
    """
    Display simple data quality notes for transparency.
    """
    with st.expander("Data Quality and Column Debugging"):
        st.write("Available columns after normalization:")
        st.write(list(df.columns))

        missing_values = (
            df.isna()
            .sum()
            .reset_index()
        )
        missing_values.columns = ["column", "missing_values"]
        missing_values = missing_values[
            missing_values["missing_values"] > 0
        ].sort_values("missing_values", ascending=False)

        if missing_values.empty:
            st.success("No missing values detected after loading and cleaning.")
        else:
            st.write("Columns with missing values:")
            st.dataframe(missing_values, use_container_width=True, hide_index=True)


# ============================================================
# Main App
# ============================================================

def main() -> None:
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    raw_df, load_error = load_data()

    if load_error:
        st.error(load_error)
        st.stop()

    df = clean_data(raw_df)

    if not validate_columns(df):
        st.stop()

    df = create_features(df)

    display_data_quality_notes(df)

    filtered_df = apply_filters(df)

    if filtered_df.empty:
        st.warning(
            "No records match the selected filters. Please adjust the sidebar filters."
        )
        st.stop()

    display_kpis(filtered_df)

    st.divider()

    build_charts(filtered_df)

    st.divider()

    display_tables_and_downloads(filtered_df)


if __name__ == "__main__":
    main()


# Deployment Instructions:
# 1. Push app.py, the dataset file, and requirements.txt to a GitHub repository.
# 2. Deploy the repository using Streamlit Community Cloud or Hugging Face Spaces.
# 3. Confirm the deployed app loads correctly and all dashboard filters/charts work.
