import streamlit as st
import pandas as pd
import datacompy
import plotly.express as px
from datetime import datetime
import requests
from io import BytesIO
import unicodedata
import re

# Page configuration
st.set_page_config(
    page_title="✨ Dataset Analyzer Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .css-1d391kg {
        padding: 2rem;
        border-radius: 1rem;
        background: #f8f9fa;
    }
    .st-emotion-cache-1wbqy5l {
        border: 1px solid #e9ecef;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🔍 Dataset Analyzer Pro")
st.markdown("### Compare, Analyze, and Visualize Your Data")

# Sidebar
with st.sidebar:
    # Try to load the Streamlit logo from the web; if it fails (network or blocked),
    # fall back to a simple text header to avoid client-side image errors.
    logo_url = "https://raw.githubusercontent.com/streamlit/docs/main/public/logos/streamlit-mark-color.png"
    try:
        resp = requests.get(logo_url, timeout=3)
        if resp.status_code == 200 and resp.content:
            st.image(BytesIO(resp.content), width=100)
        else:
            st.markdown("### Dataset Analyzer Pro")
    except Exception:
        st.markdown("### Dataset Analyzer Pro")
    st.header("Settings")
    tolerance = st.slider("Numerical Tolerance", 0.0, 1.0, 0.0, 0.01)
    show_viz = st.checkbox("Show Visualizations", True)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.info("📊 Dataset A")
    file_a = st.file_uploader("Upload your first dataset", type=["csv"])
    if file_a:
        df_a = pd.read_csv(file_a, low_memory=False)
        # Normalize column names: ensure they are strings and strip surrounding whitespace
        def _clean_col_name(s):
            # Normalize unicode, remove BOM and non-breaking spaces, collapse whitespace, strip
            s = str(s)
            s = unicodedata.normalize('NFKC', s)
            s = s.replace('\ufeff', '')
            s = s.replace('\xa0', ' ')
            s = re.sub(r"\s+", ' ', s)
            return s.strip()

        df_a.columns = [_clean_col_name(c) for c in df_a.columns]
        with st.expander("Preview Dataset A", expanded=True):
            st.dataframe(df_a.head())
            st.caption(f"Total rows: {len(df_a)} | Total columns: {len(df_a.columns)}")

            # Quick stats for numeric columns
            num_cols = df_a.select_dtypes(include=['int64', 'float64']).columns
            if len(num_cols) > 0 and show_viz:
                col_to_viz = st.selectbox("Select column to visualize (Dataset A)", num_cols, key="viz_a_select")
                fig = px.histogram(df_a, x=col_to_viz, title=f"Distribution of {col_to_viz}")
                st.plotly_chart(fig, width='stretch', key="viz_a_chart")

with col2:
    st.info("📈 Dataset B")
    file_b = st.file_uploader("Upload your second dataset", type=["csv"])
    if file_b:
        df_b = pd.read_csv(file_b, low_memory=False)
        # Normalize column names: ensure they are strings and strip surrounding whitespace
        df_b.columns = [_clean_col_name(c) for c in df_b.columns]
        with st.expander("Preview Dataset B", expanded=True):
            st.dataframe(df_b.head())
            st.caption(f"Total rows: {len(df_b)} | Total columns: {len(df_b.columns)}")

            # Quick stats for numeric columns
            num_cols = df_b.select_dtypes(include=['int64', 'float64']).columns
            if len(num_cols) > 0 and show_viz:
                col_to_viz = st.selectbox("Select column to visualize (Dataset B)", num_cols, key="viz_b_select")
                fig = px.histogram(df_b, x=col_to_viz, title=f"Distribution of {col_to_viz}")
                st.plotly_chart(fig, width='stretch', key="viz_b_chart")

if file_a and file_b:
    st.markdown("---")
    st.header("🔄 Comparison Analysis")

    # Get common columns
    common_cols = list(set(df_a.columns) & set(df_b.columns))

    if not common_cols:
        st.error("❌ No common columns found between the datasets!")
    else:
        # Column selection
        col1, col2 = st.columns([2, 1])
        with col1:
            join_cols = st.multiselect(
                "🔑 Select Key Columns for Comparison",
                common_cols,
                default=[common_cols[0]]
            )

        with col2:
            st.info(f"📝 {len(common_cols)} common columns found")

        if join_cols:
            try:
                # Prepare DataFrames for comparison
                df_a_prep = df_a.copy()
                df_b_prep = df_b.copy()

                # Resolve join columns: ensure the selected join columns exist in both dataframes.
                resolved_join = []
                missing_keys = []
                a_cols_lower = {c.lower().strip(): c for c in df_a_prep.columns}
                b_cols_lower = {c.lower().strip(): c for c in df_b_prep.columns}

                for key in join_cols:
                    key_norm = key.lower().strip()
                    if key in df_a_prep.columns and key in df_b_prep.columns:
                        resolved_join.append(key)
                    else:
                        # try case-insensitive match
                        a_match = a_cols_lower.get(key_norm)
                        b_match = b_cols_lower.get(key_norm)
                        if a_match and b_match:
                            # Rename the column in df_b_prep to match df_a_prep
                            df_b_prep.rename(columns={b_match: a_match}, inplace=True)
                            resolved_join.append(a_match)
                        else:
                            missing_keys.append(key)

                if missing_keys:
                    st.error(f"Key column(s) not found in both datasets: {missing_keys}")
                    raise ValueError(f"Missing key columns: {missing_keys}")

                # Recalculate common columns after renaming
                common_cols = list(set(df_a_prep.columns) & set(df_b_prep.columns))

                # Convert join columns to string to prevent merge/datacompy errors
                for col in resolved_join:
                    if col in df_a_prep.columns and col in df_b_prep.columns:
                        df_a_prep[col] = df_a_prep[col].fillna('').astype(str).str.strip()
                        df_b_prep[col] = df_b_prep[col].fillna('').astype(str).str.strip()
                    else:
                        st.error(f"Join column '{col}' not found in both datasets after preparation.")
                        raise ValueError(f"Missing join column: {col}")

                # Convert all other columns to string to ensure consistent comparison
                for col in df_a_prep.columns:
                    if col not in resolved_join:
                        df_a_prep[col] = df_a_prep[col].fillna('').astype(str).str.strip()
                for col in df_b_prep.columns:
                    if col not in resolved_join:
                        df_b_prep[col] = df_b_prep[col].fillna('').astype(str).str.strip()
                        
                

                # Create comparison
                import traceback

                # ===== KEY STATISTICS VISUALIZATION =====
                st.markdown("### 🔑 Key Column Statistics")
                
                # Collect statistics for each join column
                key_stats_a = {}
                key_stats_b = {}
                
                for col in resolved_join:
                    if col in df_a_prep.columns:
                        s_a = df_a_prep[col].astype(str)
                        key_stats_a[col] = {
                            'total_rows': len(s_a),
                            'unique_count': int(s_a[s_a != ""].nunique()),
                            'nulls': int((s_a == "").sum()),
                            'duplicates': len(s_a) - int(s_a.nunique()),
                            'non_null_sample': s_a[s_a != ""].head(5).tolist()
                        }
                    
                    if col in df_b_prep.columns:
                        s_b = df_b_prep[col].astype(str)
                        key_stats_b[col] = {
                            'total_rows': len(s_b),
                            'unique_count': int(s_b[s_b != ""].nunique()),
                            'nulls': int((s_b == "").sum()),
                            'duplicates': len(s_b) - int(s_b.nunique()),
                            'non_null_sample': s_b[s_b != ""].head(5).tolist()
                        }
                
                # Display statistics in organized columns for each key
                for col in resolved_join:
                    st.markdown(f"#### 📊 Key Column: `{col}`")
                    
                    col_left, col_right = st.columns(2)
                    
                    with col_left:
                        st.markdown("**📁 Dataset A**")
                        if col in key_stats_a:
                            stats = key_stats_a[col]
                            
                            # Metrics in sub-columns
                            m1, m2, m3 = st.columns(3)
                            with m1:
                                st.metric("Total Rows", f"{stats['total_rows']:,}")
                            with m2:
                                st.metric("Unique Values", f"{stats['unique_count']:,}")
                            with m3:
                                st.metric("Nulls/Empty", f"{stats['nulls']:,}")
                            
                            # Duplicates indicator
                            if stats['duplicates'] > 0:
                                st.warning(f"⚠️ {stats['duplicates']:,} duplicate rows detected")
                            else:
                                st.success("✅ No duplicates found")
                            
                            # Sample values
                            if stats['non_null_sample']:
                                with st.expander("🔍 Sample Values"):
                                    for i, val in enumerate(stats['non_null_sample'][:5], 1):
                                        st.text(f"{i}. {val}")
                        else:
                            st.error(f"Column '{col}' not found in Dataset A")
                    
                    with col_right:
                        st.markdown("**📁 Dataset B**")
                        if col in key_stats_b:
                            stats = key_stats_b[col]
                            
                            # Metrics in sub-columns
                            m1, m2, m3 = st.columns(3)
                            with m1:
                                st.metric("Total Rows", f"{stats['total_rows']:,}")
                            with m2:
                                st.metric("Unique Values", f"{stats['unique_count']:,}")
                            with m3:
                                st.metric("Nulls/Empty", f"{stats['nulls']:,}")
                            
                            # Duplicates indicator
                            if stats['duplicates'] > 0:
                                st.warning(f"⚠️ {stats['duplicates']:,} duplicate rows detected")
                            else:
                                st.success("✅ No duplicates found")
                            
                            # Sample values
                            if stats['non_null_sample']:
                                with st.expander("🔍 Sample Values"):
                                    for i, val in enumerate(stats['non_null_sample'][:5], 1):
                                        st.text(f"{i}. {val}")
                        else:
                            st.error(f"Column '{col}' not found in Dataset B")
                    
                    st.markdown("---")
                
                # Visual comparison chart if there's only one key column
                if len(resolved_join) == 1 and show_viz:
                    col = resolved_join[0]
                    if col in key_stats_a and col in key_stats_b:
                        st.markdown("#### 📈 Visual Comparison")
                        
                        comparison_data = pd.DataFrame({
                            'Metric': ['Total Rows', 'Unique Values', 'Nulls/Empty', 'Duplicates'],
                            'Dataset A': [
                                key_stats_a[col]['total_rows'],
                                key_stats_a[col]['unique_count'],
                                key_stats_a[col]['nulls'],
                                key_stats_a[col]['duplicates']
                            ],
                            'Dataset B': [
                                key_stats_b[col]['total_rows'],
                                key_stats_b[col]['unique_count'],
                                key_stats_b[col]['nulls'],
                                key_stats_b[col]['duplicates']
                            ]
                        })
                        
                        fig = px.bar(
                            comparison_data,
                            x='Metric',
                            y=['Dataset A', 'Dataset B'],
                            barmode='group',
                            title=f"Key Statistics Comparison: {col}",
                            labels={'value': 'Count', 'variable': 'Dataset'},
                            color_discrete_map={'Dataset A': '#636EFA', 'Dataset B': '#EF553B'}
                        )
                        st.plotly_chart(fig, use_container_width=True, key="key_stats_chart")
                
                st.markdown("---")

                try:
                    comp = datacompy.Compare(
                        df_a_prep,
                        df_b_prep,
                        join_columns=resolved_join,
                        abs_tol=tolerance,
                        rel_tol=tolerance,
                        ignore_spaces=True,
                        ignore_case=True
                    )
                except Exception as _e:
                    st.error("Error creating datacompy.Compare")
                    st.exception(_e)
                    st.code(traceback.format_exc())
                    raise

                # Results in cards
                st.markdown("### 📊 Comparison Results")
                c1, c2, c3, c4, c5 = st.columns(5)

                # Basic statistics
                total_rows_a = len(df_a_prep)
                total_rows_b = len(df_b_prep)

                # Build an inner-joined DataFrame on the resolved key columns to compare matched rows
                merged = df_a_prep.merge(
                    df_b_prep,
                    on=resolved_join,
                    how='inner',
                    suffixes=("_a", "_b")
                )

                # Columns to compare (exclude join/key columns). Use intersection of actual columns.
                compared_cols = [c for c in common_cols if c not in resolved_join]

                # If there are compared columns, compute per-row mismatches (any column differs)
                mismatch_count = 0
                matched_rows = merged.shape[0]
                mismatched_preview = None
                if len(compared_cols) > 0 and matched_rows > 0:
                    # start with all False (no mismatch)
                    mismatch_series = pd.Series(False, index=merged.index)

                    for col in compared_cols:
                        try:
                            # per-column safe comparison
                            col_a = f"{col}_a"
                            col_b = f"{col}_b"
                            if col_a not in merged.columns or col_b not in merged.columns:
                                # maybe the suffix names differ due to renames; skip if absent
                                continue

                            a_vals = merged[col_a]
                            b_vals = merged[col_b]

                            # Since all columns are now strings, compare as strings
                            a_str = a_vals.where(a_vals.notna(), "").astype(str).str.strip().str.lower()
                            b_str = b_vals.where(b_vals.notna(), "").astype(str).str.strip().str.lower()
                            col_mismatch = ~(a_str == b_str)

                            mismatch_series = mismatch_series | col_mismatch.fillna(False)
                        except Exception:
                            # If any column comparison fails, skip the column but log a small warning
                            # Avoid raising to keep analysis running.
                            continue

                    mismatch_count = int(mismatch_series.sum())
                    matched_rows = int((~mismatch_series).sum())

                    # Prepare a preview DataFrame of mismatched rows (keys + both versions)
                    if mismatch_count > 0:
                        keys = resolved_join
                        cols_to_show = keys[:]
                        for c in compared_cols[:10]:
                            # show up to 10 compared columns in preview to avoid huge output
                            cols_to_show.extend([f"{c}_a", f"{c}_b"]) if f"{c}_a" in merged.columns and f"{c}_b" in merged.columns else None
                        mismatched_preview = merged.loc[mismatch_series, cols_to_show].head(200)

                # Rows only in each side
                unq_a = len(comp.df1_unq_rows) if hasattr(comp, 'df1_unq_rows') else 0
                unq_b = len(comp.df2_unq_rows) if hasattr(comp, 'df2_unq_rows') else 0

                with c1:
                    match_percent = (matched_rows / total_rows_a * 100) if total_rows_a > 0 else 0
                    st.metric("Matching Rows", str(matched_rows), delta=f"{match_percent:.1f}%")
                with c2:
                    st.metric("Rows Only in A", str(unq_a))
                with c3:
                    st.metric("Rows Only in B", str(unq_b))
                with c4:
                    st.metric("Common Columns", str(len(common_cols)))
                with c5:
                    st.metric("Mismatched Rows", str(mismatch_count))

                # Detailed Analysis
                tabs = st.tabs(["📝 Report", "🔍 Null Analysis", "📊 Column Stats", "🔎 Mismatches"])

                with tabs[0]:
                    st.code(comp.report())

                with tabs[1]:
                    col1, col2 = st.columns(2)
                    with col1:
                        null_a = df_a.isna().sum()
                        if null_a.sum() > 0:
                            st.warning("Null Values in Dataset A")
                            st.dataframe(null_a[null_a > 0])
                        else:
                            st.success("No null values in Dataset A")

                    with col2:
                        null_b = df_b.isna().sum()
                        if null_b.sum() > 0:
                            st.warning("Null Values in Dataset B")
                            st.dataframe(null_b[null_b > 0])
                        else:
                            st.success("No null values in Dataset B")

                with tabs[2]:
                    # Statistical comparison of numeric columns
                    numeric_cols = list(set(df_a.select_dtypes(include=['int64', 'float64']).columns) &
                                     set(df_b.select_dtypes(include=['int64', 'float64']).columns))

                    if numeric_cols and show_viz:
                        selected_col = st.selectbox("Select column for statistical comparison", numeric_cols, key="stats_select")
                        col1, col2 = st.columns(2)

                        with col1:
                            title_a = "Distribution in Dataset A: " + selected_col
                            fig = px.box(df_a, y=selected_col, title=title_a)
                            st.plotly_chart(fig, width='stretch', key="stats_a_chart")
                            
                        with col2:
                            title_b = "Distribution in Dataset B: " + selected_col
                            fig = px.box(df_b, y=selected_col, title=title_b)
                            st.plotly_chart(fig, width='stretch', key="stats_b_chart")

                # Export options
                st.markdown("---")
                if st.button("📥 Export Comparison Report"):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    report_filename = f"comparison_report_{timestamp}.txt"
                    with open(report_filename, "w") as f:
                        f.write(comp.report())
                    st.success(f"Report exported to {report_filename}")

                # Show mismatched preview in the last tab
                with tabs[3]:
                    if mismatched_preview is not None and not mismatched_preview.empty:
                        st.write(f"Showing top {len(mismatched_preview)} mismatched rows (keys + compared columns)")
                        st.dataframe(mismatched_preview)
                        if st.button("📥 Export mismatched rows (CSV)"):
                            out_name = f"mismatched_rows_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                            mismatched_preview.to_csv(out_name, index=False)
                            st.success(f"Exported to {out_name}")
                    else:
                        st.info("No mismatched rows detected for the selected key(s) and compared columns.")
            except Exception as e:
                # Provide a clearer hint when a join/key column is missing or trimmed
                err_msg = str(e)
                st.error(f"❌ Error comparing datasets: {err_msg}")
                try:
                    # show column lists to help diagnose KeyError on join columns
                    if 'file_a' in locals() and 'file_b' in locals():
                        st.info("Column names in Dataset A (post-normalization):")
                        # show reprs and lengths to reveal hidden characters
                        st.write([repr(c) for c in df_a.columns])
                        st.write([{'name': c, 'len': len(c)} for c in df_a.columns])
                        st.info("Column names in Dataset B (post-normalization):")
                        st.write([repr(c) for c in df_b.columns])
                        st.write([{'name': c, 'len': len(c)} for c in df_b.columns])
                except Exception:
                    pass
