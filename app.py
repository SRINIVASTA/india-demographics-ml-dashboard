import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from demographics_engine import load_all_india_matrix, generate_forecasts

st.set_page_config(page_title="India Master Demographic Analyzer", layout="wide")

# Process initial data matrices safely behind the scenes
df_base = load_all_india_matrix()

st.sidebar.header("🎛️ Dashboard Configuration")
regions_list = sorted(df_base['region'].unique())

selected_region = st.sidebar.selectbox(
    "Select Target State/UT:", regions_list, 
    index=regions_list.index("India (Total Country)"),
    key="dashboard_region_dropdown"
)

selected_model = st.sidebar.radio(
    "Machine Learning Ensemble Strategy:", 
    options=["Ridge", "Lasso", "Linear Regression"],
    help="Switch estimators to view variance across forecast matrices.",
    key="dashboard_ml_radio"
)

# Run the forecast pipeline engine
df_complete = generate_forecasts(df_base, model_type=selected_model)
available_years = sorted(list(df_complete['year'].unique()))
from_year, to_year = st.sidebar.select_slider(
    "Select Project Analysis Horizon Window:", 
    options=available_years, value=(1951, 2036),
    key="dashboard_horizon_slider"
)

st.title(f"📊 {selected_region} Demographic Dashboard ({from_year}–{to_year})")
st.caption(f"Blends tracking benchmarks with continuous **{selected_model} Engine** forecasts and 95% Confidence Bounds.")

df_display = df_complete[(df_complete['region'] == selected_region) & (df_complete['year'] >= from_year) & (df_complete['year'] <= to_year)].sort_values('year')
df_hist = df_display[df_display['data_type'] == "Government Certified Base"]
df_fore = df_display[df_display['data_type'] == "ML Forecast Projection"]

fig, ax1 = plt.subplots(figsize=(12, 5.5))
handles = []

if not df_hist.empty:
    h1, = ax1.plot(df_hist['year'], df_hist['births_millions'], color='teal', marker='o', linewidth=2, label='Births (Historical)')
    h2, = ax1.plot(df_hist['year'], df_hist['deaths_millions'], color='crimson', marker='s', linewidth=2, label='Deaths (Historical)')
    h3, = ax1.plot(df_hist['year'], df_hist['net_growth_millions'], color='forestgreen', linestyle='--', marker='^', label='Net Growth (Historical)')
    handles.extend([h1, h2, h3])

if not df_fore.empty:
    h4, = ax1.plot(df_fore['year'], df_fore['births_millions'], color='teal', linestyle=':', marker='o', alpha=0.7, label='Births (ML)')
    h5, = ax1.plot(df_fore['year'], df_fore['deaths_millions'], color='crimson', linestyle=':', marker='s', alpha=0.7, label='Deaths (ML)')
    h6, = ax1.plot(df_fore['year'], df_fore['net_growth_millions'], color='forestgreen', linestyle=':', marker='^', alpha=0.7, label='Net Growth (ML)')
    handles.extend([h4, h5, h6])

ax1.set_ylabel('Annual Events Volume (In Millions)', color='black', fontsize=11, fontweight='bold')
ax1.set_xlabel('Timeline Track Year', fontsize=11, fontweight='bold')
ax1.set_ylim(min(0, df_display['net_growth_millions'].min()) * 1.1, max(df_display['births_millions'].max(), df_display['deaths_millions'].max()) * 1.3)

ax2 = ax1.twinx()
if not df_hist.empty:
    h7, = ax2.plot(df_hist['year'], df_hist['population_millions'], color='indigo', linestyle='-', marker='d', linewidth=2, label='Population (Historical)')
    handles.append(h7)

if not df_fore.empty:
    h8, = ax2.plot(df_fore['year'], df_fore['population_millions'], color='indigo', linestyle=':', marker='d', alpha=0.7, label=f'Population ({selected_model})')
    handles.append(h8)
    ax2.fill_between(df_fore['year'], df_fore['pop_lower_ci'], df_fore['pop_upper_ci'], color='indigo', alpha=0.15, label='95% Confidence Band')

ax2.set_ylabel('Total Cumulative Population (In Millions)', color='indigo', fontsize=11, fontweight='bold')
ax2.set_ylim(0, df_display['population_millions'].max() * 1.35)

labels = [h.get_label() for h in handles]
ax1.legend(handles, labels, loc='upper left', shadow=True, fontsize=9)
ax1.grid(True, linestyle=':', alpha=0.5)
ax1.set_xticks(df_display['year'])
ax1.set_xticklabels(df_display['year'], rotation=90, ha='center') 
fig.tight_layout() 

st.pyplot(fig)

st.subheader(f"📋 Dataset Summary Grid view: {selected_region}")
st.dataframe(df_display[['year', 'population_millions', 'births_millions', 'deaths_millions', 'net_growth_millions', 'data_type']], use_container_width=True)

st.download_button(
    label="📥 Download Currently Selected Sheet Extract View as CSV",
    data=df_display.to_csv(index=False).encode('utf-8'),
    file_name=f"{selected_region.lower().replace(' ', '_')}_demographics_extract.csv",
    mime="text/csv",
    key="dashboard_download_button"
)

# ==========================================
# 📊 INTERNAL PYTEST AUTOMATION EXPANDER
# ==========================================
st.markdown("---")
st.subheader("🛠️ Developer Quality Assurance Suite")

with st.expander("🔐 Run System Validation Test Suite (Pytest Docs)"):
    st.write("Click the button below to headlessly execute your active `test_app.py` suite.")
    
    if st.button("🚀 Execute Pytest Suite Engine", key="dashboard_pytest_trigger_btn"):
        import sys
        import io
        
        with st.spinner("Running automated mathematical checks..."):
            try:
                import pytest
                captured_buffer = io.StringIO()
                sys.stdout = captured_buffer
                sys.stderr = captured_buffer
                
                exit_code = pytest.main(["test_app.py", "-v", "-p", "no:warnings"])
                
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__
                console_logs = captured_buffer.getvalue()
                
                if exit_code == 0:
                    st.success("✅ ALL CODE CHECK PASSES: Your backend math equations are stable!")
                    st.code(console_logs, language="text")
                else:
                    st.error("❌ TESTING ANOMALY SPOTTED: One of your analytical assertion blocks failed.")
                    st.code(console_logs, language="text")
                    
            except Exception as e:
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__
                st.exception(f"Critical execution error: {e}")
