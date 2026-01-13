import difflib
import re
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np


EXPECTED_COLUMNS = [
    "task_date",
    "hours_spent",
    "employee",
    "racfid",
    "rank",
    "unit",
    "remarks",
    "ticket_number",
    "ticket_type",
    "work_setup",
    "service_type",
    "type",
    "project",
    "leader_name",
]

EMPLOYEE_ROSTER_PATH = Path(__file__).parent / "files" / "employee_names_fn_ln.csv"
WORK_WEEK_HOURS = 40
HOLIDAY_DAY_HOURS = 8


@st.cache_data(show_spinner=False)
def load_employee_roster() -> list[str]:
    df = pd.read_csv(EMPLOYEE_ROSTER_PATH)
    return df["name"].dropna().astype(str).tolist()


def normalize_name(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    words = re.sub(r"\s+", " ", cleaned).split()
    if len(words) >= 2:
        return f"{words[0]} {words[-1]}"
    return " ".join(words)


def compute_compliance(
    uploaded_df: pd.DataFrame, roster: list[str]
) -> tuple[float, list[str], int, int]:
    roster_unique = [name for name in dict.fromkeys(roster)]

    uploaded_names = (
        uploaded_df["employee"].dropna().astype(str).map(normalize_name)
        if "employee" in uploaded_df.columns
        else pd.Series(dtype=str)
    )
    uploaded_normalized = {name for name in uploaded_names if name}

    unmatched: list[str] = []
    for roster_name in roster_unique:
        normalized_roster = normalize_name(roster_name)
        if not normalized_roster:
            continue
        if normalized_roster in uploaded_normalized:
            continue
        close_match = difflib.get_close_matches(
            normalized_roster,
            list(uploaded_normalized),
            n=1,
            cutoff=0.85,
        )
        if not close_match:
            unmatched.append(roster_name)

    total = len(roster_unique)
    matched = total - len(unmatched)
    compliance_pct = (matched / total * 100) if total else 0.0
    return compliance_pct, unmatched, matched, total


def main() -> None:
    st.set_page_config(page_title="Prompt Viz", layout="wide")
    st.title("Prompt Viz")
    st.caption("Upload a CSV to explore its entries.")

    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    exempt_file = st.file_uploader(
        "Upload exempt employees CSV (optional)", type="csv"
    )
    apply_holiday = st.checkbox("Apply holiday adjustment to utilization")
    holiday_date: date | None = None
    if apply_holiday:
        holiday_date = st.date_input(
            "Holiday date",
            value=date.today(),
            format="YYYY-MM-DD",
        )

    if uploaded_file is None:
        st.info("Awaiting CSV upload. The file must include the expected columns.")
        return

    df = pd.read_csv(uploaded_file)

    missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_columns:
        st.warning(
            "The uploaded file is missing expected columns: " + ", ".join(missing_columns)
        )

    exempt_normalized: set[str] = set()
    if exempt_file is not None:
        try:
            exempt_df = pd.read_csv(exempt_file)
        except Exception as exc:  # pragma: no cover - Streamlit runtime safety
            st.warning(f"Unable to read exempt CSV: {exc}")
        else:
            if "name" not in exempt_df.columns:
                st.warning("Exempt CSV missing required 'name' column; ignoring file.")
            else:
                exempt_values = (
                    exempt_df["name"].dropna().astype(str).map(normalize_name)
                )
                exempt_normalized = {
                    name for name in exempt_values.tolist() if name
                }
                if not exempt_normalized:
                    st.info("Exempt CSV did not contain any valid employee names.")

    if exempt_normalized and "employee" in df.columns:
        before_rows = len(df)
        employee_norm = df["employee"].fillna("").astype(str).map(normalize_name)
        df = df.loc[~employee_norm.isin(exempt_normalized)].copy()
        removed_rows = before_rows - len(df)
        if removed_rows:
            st.caption(
                f"Excluded {removed_rows} exempt row(s) from the uploaded dataset."
            )

    date_filter_caption: str | None = None
    if "task_date" in df.columns:
        parsed_dates = pd.to_datetime(df["task_date"], errors="coerce")
        valid_dates = parsed_dates.dropna()
        if not valid_dates.empty:
            min_date = valid_dates.min().date()
            max_date = valid_dates.max().date()
            range_value = st.date_input(
                "Task date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )
            if isinstance(range_value, tuple):
                start_date, end_date = range_value
            else:
                start_date = end_date = range_value
            if start_date > end_date:
                start_date, end_date = end_date, start_date
            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date)
            mask = parsed_dates.between(start_ts, end_ts, inclusive="both")
            kept_rows = mask.sum()
            df = df.loc[mask].copy()
            dropped = len(mask) - kept_rows
            if dropped or start_date != min_date or end_date != max_date:
                date_filter_caption = (
                    f"Filtered to {start_date:%Y-%m-%d} – {end_date:%Y-%m-%d}."
                )
                if dropped:
                    date_filter_caption += f" Removed {dropped} row(s) outside the range or without valid dates."

    st.dataframe(df)
    if date_filter_caption:
        st.caption(date_filter_caption)
    if holiday_date:
        adjusted_week = max(WORK_WEEK_HOURS - HOLIDAY_DAY_HOURS, 0)
        st.caption(
            f"Utilization capacity reduced to {adjusted_week} hours per employee due to holiday on {holiday_date:%Y-%m-%d}."
        )

    if "task_date" in df.columns:
        date_series = pd.to_datetime(df["task_date"], errors="coerce")
        valid_dates = date_series.dropna()
        if not valid_dates.empty:
            min_date = valid_dates.min()
            max_date = valid_dates.max()
            st.info(
                f"Date coverage: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
            )

    try:
        roster_names = load_employee_roster()
    except FileNotFoundError:
        st.error("Employee roster file not found. Please contact the administrator.")
        return
    except pd.errors.EmptyDataError:
        st.error("Employee roster file is empty. Please contact the administrator.")
        return
    if exempt_normalized:
        roster_names = [
            name
            for name in roster_names
            if normalize_name(name) not in exempt_normalized
        ]
    compliance_pct, missing_names, matched_count, total_roster = compute_compliance(
        df, roster_names
    )

    st.subheader("Compliance Rating")
    st.metric(
        label="Roster Coverage",
        value=f"{compliance_pct:.1f}%",
        delta=f"{matched_count}/{total_roster}"
        if total_roster
        else None,
    )

    if missing_names:
        st.dataframe(pd.DataFrame({"Employee Not Found": missing_names}))
    else:
        st.success("Every rostered employee is present in the uploaded CSV.")

    st.subheader("Man-Hour Utilization")
    if "hours_spent" in df.columns:
        hours_numeric = pd.to_numeric(df["hours_spent"], errors="coerce")
    else:
        hours_numeric = pd.Series(0.0, index=df.index)
    unit_group = (
        df.assign(hours_spent=hours_numeric)
        .groupby("unit", dropna=False)
        .agg(
            total_hours=("hours_spent", "sum"),
            employee_count=("employee", pd.Series.nunique),
        )
        .reset_index()
    )
    unit_group["total_hours"].fillna(0, inplace=True)
    unit_group["employee_count"].fillna(0, inplace=True)
    per_employee_capacity = max(
        WORK_WEEK_HOURS - (HOLIDAY_DAY_HOURS if holiday_date else 0),
        0,
    )
    unit_group["utilization_capacity"] = (
        unit_group["employee_count"] * per_employee_capacity
    )
    unit_group["utilization_pct"] = unit_group.apply(
        lambda row: (row.total_hours / row.utilization_capacity * 100)
        if row.utilization_capacity
        else 0.0,
        axis=1,
    )
    unit_group.rename(
        columns={
            "unit": "Unit",
            "total_hours": "Hours Spent",
            "employee_count": "Distinct Employees",
            "utilization_capacity": "Available Hours",
            "utilization_pct": "Utilization %",
        },
        inplace=True,
    )

    totals = {
        "Unit": "Grand Total",
        "Hours Spent": unit_group["Hours Spent"].sum(),
        "Distinct Employees": unit_group["Distinct Employees"].sum(),
        "Available Hours": unit_group["Available Hours"].sum(),
    }
    totals["Utilization %"] = (
        totals["Hours Spent"] / totals["Available Hours"] * 100
        if totals["Available Hours"]
        else 0.0
    )

    unit_group = pd.concat([unit_group, pd.DataFrame([totals])], ignore_index=True)
    st.dataframe(
        unit_group,
        column_config={
            "Hours Spent": st.column_config.NumberColumn(format="%.2f"),
            "Distinct Employees": st.column_config.NumberColumn(format="%d"),
            "Available Hours": st.column_config.NumberColumn(format="%.2f"),
            "Utilization %": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )

    st.subheader("Man-hours Spent by Work Setup")
    if "work_setup" not in df.columns:
        st.info("The uploaded file does not include a 'work_setup' column.")
        return

    work_setup_clean = df["work_setup"].fillna("Unspecified")
    
    # Total distribution summarized for the pie chart
    work_setup_totals = (
        df.assign(hours_spent=hours_numeric, work_setup=work_setup_clean)
        .groupby("work_setup", dropna=False)["hours_spent"]
        .sum()
        .reset_index()
        .sort_values("hours_spent", ascending=False)
    )

    if not work_setup_totals.empty and work_setup_totals["hours_spent"].sum() > 0:
        fig_pie = px.pie(
            work_setup_totals,
            names="work_setup",
            values="hours_spent",
            title="Total Man-Hours Distribution by Work Setup",
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie, width='stretch')

    work_setup_pivot = (
        df.assign(hours_spent=hours_numeric, work_setup=work_setup_clean)
        .groupby(["unit", "work_setup"], dropna=False)["hours_spent"]
        .sum()
        .reset_index()
        .pivot_table(
            index="unit",
            columns="work_setup",
            values="hours_spent",
            fill_value=0.0,
        )
    )

    if not work_setup_pivot.empty:
        row_totals = work_setup_pivot.sum(axis=1)
        percentage_by_setup = work_setup_pivot.div(row_totals.replace(0, pd.NA), axis=0) * 100
        percentage_by_setup.fillna(0.0, inplace=True)
        percentage_by_setup = percentage_by_setup.round(1)

        percentage_display = percentage_by_setup.reset_index().rename(columns={"unit": "Unit"})
        percentage_columns = [col for col in percentage_display.columns if col != "Unit"]
        column_config = {
            col: st.column_config.NumberColumn(format="%.1f%%") for col in percentage_columns
        }
        st.write("Percentage breakdown per unit:")
        st.dataframe(percentage_display, column_config=column_config)

        stacked_data = (
            percentage_by_setup.reset_index()
            .rename(columns={"unit": "Unit"})
            .melt(id_vars="Unit", var_name="Work Setup", value_name="Percentage")
        )

        if stacked_data["Percentage"].sum() > 0:
            fig_work_setup = px.bar(
                stacked_data,
                x="Unit",
                y="Percentage",
                color="Work Setup",
                title="Work Setup Share per Unit",
                barmode="stack",
            )
            fig_work_setup.update_layout(yaxis_title="Percentage", xaxis_title="Unit")
            st.plotly_chart(fig_work_setup, width='stretch')

    st.subheader("Man-hours Spent by Task Type")
    if "type" not in df.columns:
        st.info("The uploaded file does not include a 'type' column.")
        return

    task_type_clean = df["type"].fillna("Unspecified")
    task_type_totals = (
        df.assign(hours_spent=hours_numeric, type=task_type_clean)
        .groupby("type", dropna=False)["hours_spent"]
        .sum()
        .reset_index()
    )

    if task_type_totals.empty:
        st.info("No man-hour data available to summarize by task type.")
        return

    task_type_totals.rename(
        columns={"type": "Task Type", "hours_spent": "Hours Spent"},
        inplace=True,
    )
    total_hours = task_type_totals["Hours Spent"].sum()
    if total_hours == 0:
        task_type_totals["Share %"] = 0.0
    else:
        task_type_totals["Share %"] = (
            task_type_totals["Hours Spent"] / total_hours * 100
        ).round(1)

    st.dataframe(
        task_type_totals,
        column_config={
            "Hours Spent": st.column_config.NumberColumn(format="%.2f"),
            "Share %": st.column_config.NumberColumn(format="%.1f%%"),
        },
        width='stretch',
    )

    fig_task = px.pie(
        task_type_totals,
        names="Task Type",
        values="Hours Spent",
        title="Man-Hours Distribution by Task Type",
    )
    fig_task.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_task, width='stretch')

    st.subheader("Entries by Ticket Type")
    if "type" not in df.columns:
        st.info("The uploaded file does not include a 'type' column.")
        return

    ticket_entries = df[df["type"].astype(str).str.lower() == "ticket"]
    if ticket_entries.empty:
        st.info("No entries tagged as ticket task type.")
        return

    if "ticket_type" not in ticket_entries.columns:
        st.info("The uploaded file does not include a 'ticket_type' column.")
        return

    ticket_summary = (
        ticket_entries.assign(ticket_type=ticket_entries["ticket_type"].fillna("Unspecified"))
        .groupby("ticket_type")
        .agg(entry_count=("ticket_type", "count"), hours_spent=("hours_spent", "sum"))
        .reset_index()
        .sort_values("entry_count", ascending=False)
    )

    ticket_summary.rename(
        columns={
            "ticket_type": "Ticket Type",
            "entry_count": "Entries",
            "hours_spent": "Hours Spent",
        },
        inplace=True,
    )

    st.dataframe(
        ticket_summary,
        column_config={
            "Entries": st.column_config.NumberColumn(format="%d"),
            "Hours Spent": st.column_config.NumberColumn(format="%.2f"),
        },
        width='stretch',
    )

    # fig_ticket = px.pie(
    #     ticket_summary,
    #     names="Ticket Type",
    #     values="Entries",
    #     title="Ticket Entries Distribution",
    # )
    # fig_ticket.update_traces(textposition="inside", textinfo="percent+label")
    # st.plotly_chart(fig_ticket, width='stretch')

    st.subheader("Sum of Hours Spent by Ticket Type")
    ticket_hours = ticket_summary.copy()
    if ticket_hours["Hours Spent"].sum() == 0:
        st.info("Ticket entries have zero recorded hours spent.")
    else:
        ticket_hours["Share %"] = (
            ticket_hours["Hours Spent"] / ticket_hours["Hours Spent"].sum() * 100
        ).round(1)
        st.dataframe(
            ticket_hours,
            column_config={
                "Entries": st.column_config.NumberColumn(format="%d"),
                "Hours Spent": st.column_config.NumberColumn(format="%.2f"),
                "Share %": st.column_config.NumberColumn(format="%.1f%%"),
            },
            width='stretch',
        )

        fig_ticket_hours = px.pie(
            ticket_hours,
            names="Ticket Type",
            values="Hours Spent",
            title="Man-Hours Allocation by Ticket Type",
        )
        fig_ticket_hours.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_ticket_hours, width='stretch')

    st.markdown("---")
    st.subheader("Deep Dive: Solution Delivery Units Analysis")
    target_units = ["Solution Delivery - Consumer", "Solution Delivery - Corporate"]
    
    task_cols = st.columns(2)
    for idx, unit_name in enumerate(target_units):
        unit_df = df[df["unit"] == unit_name].copy()
        if unit_df.empty:
            task_cols[idx].info(f"No records found for {unit_name}")
            continue
            
        unit_df["hours_spent"] = pd.to_numeric(unit_df["hours_spent"], errors="coerce")
        task_dist = (
            unit_df.groupby("type")["hours_spent"]
            .sum()
            .reset_index()
            .sort_values("hours_spent", ascending=False)
        )
        
        if task_dist["hours_spent"].sum() > 0:
            fig_sd_task = px.pie(
                task_dist,
                names="type",
                values="hours_spent",
                title=f"Task Type: {unit_name}",
            )
            fig_sd_task.update_traces(textposition="inside", textinfo="percent+label")
            task_cols[idx].plotly_chart(fig_sd_task, use_container_width=True)

    ticket_cols = st.columns(2)
    for idx, unit_name in enumerate(target_units):
        unit_df = df[df["unit"] == unit_name].copy()
        ticket_only = unit_df[unit_df["type"].astype(str).str.lower() == "ticket"].copy()
        
        if ticket_only.empty:
            ticket_cols[idx].info(f"No ticket tasks for {unit_name}")
            continue
            
        ticket_only["hours_spent"] = pd.to_numeric(ticket_only["hours_spent"], errors="coerce")
        ticket_dist = (
            ticket_only.groupby("ticket_type")["hours_spent"]
            .sum()
            .reset_index()
            .sort_values("hours_spent", ascending=False)
        )
        
        if ticket_dist["hours_spent"].sum() > 0:
            # Using a donut chart style for variety and clarity if many tickets exist
            fig_sd_ticket = px.pie(
                ticket_dist,
                names="ticket_type",
                values="hours_spent",
                title=f"Ticket Type: {unit_name}",
                hole=0.4
            )
            fig_sd_ticket.update_traces(textposition="inside", textinfo="percent+label")
            ticket_cols[idx].plotly_chart(fig_sd_ticket, use_container_width=True)

    st.markdown("---")
    st.subheader("Ticket Type Hours by Unit")
    if "unit" not in ticket_entries.columns:
        st.info("The uploaded file does not include a 'unit' column for ticket entries.")
        return

    ticket_unit_pivot = (
        ticket_entries.assign(
            hours_spent=pd.to_numeric(ticket_entries["hours_spent"], errors="coerce"),
            ticket_type=ticket_entries["ticket_type"].fillna("Unspecified"),
            unit=ticket_entries["unit"].fillna("Unspecified"),
        )
        .pivot_table(
            index="unit",
            columns="ticket_type",
            values="hours_spent",
            aggfunc="sum",
            fill_value=0.0,
        )
    )

    if ticket_unit_pivot.empty:
        st.info("No ticket data available to summarize by unit.")
        return

    ticket_unit_pct = ticket_unit_pivot.div(
        ticket_unit_pivot.sum(axis=1).replace(0, pd.NA), axis=0
    ) * 100
    ticket_unit_pct.fillna(0.0, inplace=True)
    ticket_unit_pct = ticket_unit_pct.round(1)

    ticket_unit_display = ticket_unit_pct.reset_index().rename(columns={"unit": "Unit"})
    ticket_unit_columns = [col for col in ticket_unit_display.columns if col != "Unit"]
    ticket_unit_config = {
        col: st.column_config.NumberColumn(format="%.1f%%")
        for col in ticket_unit_columns
    }

    st.dataframe(
        ticket_unit_display,
        column_config=ticket_unit_config,
        width='stretch',
    )

    st.subheader("PROMPT Entry Volume vs Hours by Unit")
    if "unit" not in df.columns or "task_date" not in df.columns:
        st.info("The uploaded file must include both 'unit' and 'task_date' columns for this analysis.")
    else:
        productivity_df = (
            df.assign(
                hours_spent=hours_numeric,
                unit=df["unit"].fillna("").astype(str).str.strip(),
                task_date=pd.to_datetime(df["task_date"], errors="coerce"),
            )
            .loc[lambda d: d["unit"].ne("")]
            .loc[lambda d: d["task_date"].notna()]
        )

        if productivity_df.empty:
            st.info("No records with both unit and task_date are available for productivity analysis.")
        else:
            volume_summary = (
                productivity_df.groupby("unit", dropna=False)
                .agg(
                    ticket_count=("task_date", "count"),
                    hours_spent=("hours_spent", "sum"),
                )
                .reset_index()
                .rename(
                    columns={
                        "unit": "Unit",
                        "ticket_count": "PROMPT Entry Count",
                        "hours_spent": "Hours Spent",
                    }
                )
            )

            st.dataframe(
                volume_summary,
                column_config={
                    "Ticket Count": st.column_config.NumberColumn(format="%d"),
                    "Hours Spent": st.column_config.NumberColumn(format="%.2f"),
                },
                width='stretch',
            )

            if volume_summary["PROMPT Entry Count"].nunique() < 2 or len(volume_summary) < 2:
                st.info("Not enough variance in ticket counts to fit a regression model.")
            else:
                x = volume_summary["PROMPT Entry Count"].astype(float).to_numpy()
                y = volume_summary["Hours Spent"].astype(float).to_numpy()

                if np.allclose(x, x[0]):
                    st.info("PROMPT entry counts are identical across units; regression fit is not meaningful.")
                else:
                    slope, intercept = np.polyfit(x, y, 1)
                    predicted = intercept + slope * x
                    residuals = y - predicted

                    analysis_df = volume_summary.copy()
                    analysis_df["Predicted Hours"] = predicted
                    analysis_df["Residual"] = residuals
                    analysis_df["Residual %"] = np.where(
                        predicted != 0,
                        residuals / predicted * 100,
                        np.nan,
                    )

                    tolerance = max(np.std(residuals) * 0.5, 1.0)

                    def classify_residual(value: float) -> str:
                        if value > tolerance:
                            return "Positive (Above Trend)"
                        if value < -tolerance:
                            return "Negative (Below Trend)"
                        return "On Trend"

                    analysis_df["Status"] = analysis_df["Residual"].apply(classify_residual)

                    residual_view = analysis_df[
                        [
                            "Unit",
                            "PROMPT Entry Count",
                            "Hours Spent",
                            "Predicted Hours",
                            "Residual",
                            "Residual %",
                            "Status",
                        ]
                    ]

                    st.dataframe(
                        residual_view,
                        column_config={
                            "PROMPT Entry Count": st.column_config.NumberColumn(format="%d"),
                            "Hours Spent": st.column_config.NumberColumn(format="%.2f"),
                            "Predicted Hours": st.column_config.NumberColumn(format="%.2f"),
                            "Residual": st.column_config.NumberColumn(format="%.2f"),
                            "Residual %": st.column_config.NumberColumn(format="%.1f%%"),
                        },
                        width='stretch',
                    )
                    st.caption(
                        "Positive residuals indicate departments logging more hours than the model predicts; negative residuals indicate fewer hours than expected."
                    )
                    st.caption(
                        "Residuals help us find departments/employees whose results differ from expectations. A high or low residual doesn’t automatically mean good or bad performance—it simply points to areas that may need a closer look."
                    )
                    st.info(
                        f"Regression Model: Hours Spent = {intercept:.2f} + {slope:.2f} * PROMPT Entry Count"
                    )

                    line_x = np.linspace(x.min(), x.max(), 100)
                    line_y = intercept + slope * line_x
                    line_df = pd.DataFrame(
                        {
                            "PROMPT Entry Count": line_x,
                            "Predicted Hours": line_y,
                        }
                    )

                    fig_reg = px.scatter(
                        volume_summary,
                        x="PROMPT Entry Count",
                        y="Hours Spent",
                        color="Unit",
                        title="Linear Fit: Hours Spent vs Ticket Volume",
                        hover_data={"Unit": True, "PROMPT Entry Count": True, "Hours Spent": ":.2f"},
                    )
                    line_fig = px.line(
                        line_df,
                        x="PROMPT Entry Count",
                        y="Predicted Hours",
                        labels={"Predicted Hours": "Predicted Hours"},
                    )
                    for index, trace in enumerate(line_fig.data):
                        trace.name = "Regression Line"
                        trace.showlegend = index == 0
                        trace.line.color = "#2f4f4f"
                        fig_reg.add_trace(trace)
                    fig_reg.update_layout(showlegend=True)
                    st.plotly_chart(fig_reg, width='stretch')

    st.subheader("Top Projects by Hours")
    if "project" not in df.columns:
        st.info("The uploaded file does not include a 'project' column.")
        return

    project_filtered = (
        df.assign(
            hours_spent=hours_numeric,
            project=df["project"].fillna("").astype(str).str.strip(),
        )
        .loc[lambda d: d["project"].str.len() > 0]
        .loc[lambda d: d["project"].str.lower() != "unspecified"]
    )

    if project_filtered.empty:
        st.info("No project-tagged entries found in the uploaded file.")
        return

    project_summary = (
        project_filtered.groupby("project", dropna=False)
        .agg(
            total_hours=("hours_spent", "sum"),
            employee_count=("employee", pd.Series.nunique),
        )
        .reset_index()
        .sort_values("total_hours", ascending=False)
        .head(10)
    )

    project_summary.rename(
        columns={
            "project": "Project",
            "total_hours": "Hours Spent",
            "employee_count": "Distinct Employees",
        },
        inplace=True,
    )

    st.dataframe(
        project_summary,
        column_config={
            "Hours Spent": st.column_config.NumberColumn(format="%.2f"),
            "Distinct Employees": st.column_config.NumberColumn(format="%d"),
        },
        width='stretch',
    )


if __name__ == "__main__":
    main()
