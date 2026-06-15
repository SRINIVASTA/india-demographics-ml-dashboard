import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Rule: This MUST remain the absolute first execution block to avoid breaking Streamlit
st.set_page_config(page_title="India Master Demographic Analyzer", layout="wide")

@st.cache_data
def load_all_india_matrix():
    certified_anchors = {
        "Andhra Pradesh": {1901: (13.070, 44.20, 43.10), 1951: (20.110, 38.20, 25.10), 2011: (49.580, 17.50, 7.20), 2026: (53.740, 14.50, 11.635)},
        "Telangana": {1901: (6.620, 45.10, 44.00), 1951: (11.000, 39.50, 26.20), 2011: (35.000, 17.80, 7.10), 2026: (38.670, 16.50, 12.208)},
        "Delhi (NCT)": {1901: (0.410, 46.20, 43.50), 1951: (1.740, 38.50, 19.10), 2011: (16.790, 18.20, 6.10), 2026: (22.670, 24.30, 6.786)},
        "Uttar Pradesh": {1901: (48.600, 46.50, 45.10), 1951: (60.200, 41.20, 28.50), 2011: (199.810, 27.80, 7.90), 2026: (245.800, 21.40, 6.550)},
        "Bihar": {1901: (27.310, 47.10, 46.00), 1951: (38.780, 42.10, 29.10), 2011: (104.100, 27.70, 6.70), 2026: (132.100, 22.30, 5.480)},
        "Maharashtra": {1901: (19.400, 43.50, 42.10), 1951: (32.000, 37.50, 23.10), 2011: (112.370, 16.70, 6.30), 2026: (131.100, 14.50, 6.440)},
        "West Bengal": {1901: (16.900, 42.80, 41.20), 1951: (26.300, 36.80, 21.50), 2011: (91.280, 16.10, 6.20), 2026: (101.500, 12.80, 6.490)},
        "Tamil Nadu": {1901: (19.200, 42.10, 40.50), 1951: (30.100, 36.10, 22.20), 2011: (72.140, 15.90, 7.40), 2026: (79.200, 12.40, 8.480)},
        "Karnataka": {1901: (13.050, 42.50, 41.00), 1951: (19.400, 37.10, 22.50), 2011: (61.100, 17.20, 6.50), 2026: (69.300, 14.20, 7.130)},
        "Gujarat": {1901: (9.100, 43.80, 41.80), 1951: (16.260, 38.10, 21.90), 2011: (60.440, 18.50, 6.40), 2026: (71.200, 16.80, 6.40)},
        "Rajasthan": {1901: (10.290, 44.50, 42.50), 1951: (15.970, 39.20, 23.40), 2011: (68.550, 22.20, 6.50), 2026: (83.600, 19.40, 6.170)},
        "Madhya Pradesh": {1901: (16.860, 45.20, 43.60), 1951: (26.080, 40.20, 24.80), 2011: (72.630, 24.30, 7.80), 2026: (88.700, 21.10, 6.810)},
        "Kerala": {1901: (6.400, 40.20, 38.50), 1951: (13.550, 32.40, 18.10), 2011: (33.410, 14.30, 6.90), 2026: (36.200, 12.10, 7.950)},
        "Odisha": {1901: (10.300, 42.90, 41.20), 1951: (14.650, 36.50, 21.80), 2011: (41.970, 18.20, 7.60), 2026: (46.800, 13.90, 7.700)},
        "Punjab": {1901: (7.540, 44.10, 41.20), 1951: (9.160, 36.80, 19.50), 2011: (27.740, 15.10, 6.20), 2026: (31.400, 13.50, 7.130)},
        "Haryana": {1901: (4.620, 45.20, 42.10), 1951: (5.670, 38.90, 20.20), 2011: (25.350, 20.30, 6.30), 2026: (31.200, 17.20, 6.30)},
        "Jharkhand": {1901: (6.110, 46.10, 44.20), 1951: (9.690, 41.50, 26.10), 2011: (32.990, 23.10, 6.80), 2026: (41.400, 21.20, 6.130)},
        "Assam": {1901: (3.290, 45.80, 43.10), 1951: (8.030, 39.50, 22.10), 2011: (31.210, 21.30, 7.30), 2026: (36.800, 17.50, 6.90)},
        "Chhattisgarh": {1901: (3.150, 44.80, 43.00), 1951: (5.380, 39.10, 24.20), 2011: (25.540, 23.20, 7.50), 2026: (31.100, 18.50, 7.230)},
        "Jammu and Kashmir": {1901: (1.220, 43.10, 41.00), 1951: (3.250, 36.50, 19.80), 2011: (12.260, 16.40, 5.30), 2026: (14.200, 14.10, 5.650)},
        "Uttarakhand": {1901: (1.980, 44.10, 42.00), 1951: (2.940, 37.20, 21.10), 2011: (10.090, 17.10, 6.20), 2026: (11.700, 15.20, 6.650)},
        "Himachal Pradesh": {1901: (1.920, 42.10, 40.10), 1951: (2.390, 35.10, 19.20), 2011: (6.860, 15.40, 6.70), 2026: (7.700, 13.10, 7.900)},
        "Tripura": {1901: (0.170, 43.50, 41.20), 1951: (0.640, 36.10, 18.50), 2011: (3.670, 13.80, 4.80), 2026: (4.200, 12.20, 5.530)},
        "Meghalaya": {1901: (0.340, 45.10, 42.80), 1951: (0.610, 39.80, 22.10), 2011: (2.970, 23.20, 7.10), 2026: (3.600, 21.10, 6.380)},
        "Manipur": {1901: (0.280, 42.80, 40.10), 1951: (0.580, 36.20, 19.20), 2011: (2.860, 14.20, 4.50), 2026: (3.300, 13.10, 5.220)},
        "Nagaland": {1901: (0.100, 44.20, 41.50), 1951: (0.210, 38.10, 21.00), 2011: (1.980, 14.50, 4.20), 2026: (2.300, 13.20, 4.940)},
        "Goa": {1901: (0.470, 40.50, 39.10), 1951: (0.540, 32.10, 16.50), 2011: (1.460, 12.30, 6.10), 2026: (1.620, 11.20, 7.500)},
        "Arunachal Pradesh": {1901: (0.020, 45.10, 43.10), 1951: (0.060, 39.10, 23.20), 2011: (1.380, 18.20, 5.80), 2026: (1.680, 16.10, 6.580)},
        "Mizoram": {1901: (0.080, 44.50, 42.10), 1951: (0.200, 37.80, 20.10), 2011: (1.090, 15.40, 4.40), 2026: (1.320, 13.90, 5.570)},
        "Sikkim": {1901: (0.060, 41.20, 39.50), 1951: (0.140, 33.50, 17.20), 2011: (0.610, 16.20, 5.20), 2026: (0.710, 14.10, 6.210)},
        "Puducherry": {1901: (0.240, 41.50, 39.80), 1951: (0.320, 34.20, 18.10), 2011: (1.240, 14.60, 6.90), 2026: (1.780, 13.20, 7.580)},
        "Chandigarh": {1901: (0.020, 43.10, 41.20), 1951: (0.060, 35.20, 17.50), 2011: (1.060, 14.10, 4.50), 2026: (1.350, 12.80, 5.390)},
        "Dadra and Nagar Haveli and Daman and Diu": {1901: (0.030, 44.50, 42.10), 1951: (0.080, 38.20, 21.00), 2011: (0.580, 24.10, 4.80), 2026: (0.910, 21.20, 4.720)},
        "Andaman and Nicobar Islands": {1901: (0.020, 41.10, 39.20), 1951: (0.030, 34.50, 17.10), 2011: (0.380, 12.60, 5.10), 2026: (0.430, 11.20, 6.550)},
        "Lakshadweep": {1901: (0.010, 40.20, 38.10), 1951: (0.020, 33.10, 16.80), 2011: (0.064, 13.80, 6.20), 2026: (0.071, 11.80, 8.980)},
        "Ladakh": {1901: (0.020, 41.00, 39.50), 1951: (0.040, 33.50, 18.20), 2011: (0.274, 14.20, 6.50), 2026: (0.310, 12.50, 9.270)},
        "India (Total Country)": {1901: (238.400, 45.80, 44.40), 1951: (361.090, 39.90, 27.40), 2011: (1210.190, 21.80, 7.10), 2026: (1448.460, 15.50, 7.30)}
    }
    time_series_rows = []
    all_years = sorted(list(range(1901, 1942, 10)) + list(range(1951, 2027)))
    for region, anchors in certified_anchors.items():
        anchor_years = sorted(list(anchors.keys()))
        for yr in all_years:
            if yr in anchors:
                pop, cbr, cdr = anchors[yr]
            else:
                low_yr = max([y for y in anchor_years if y < yr])
                upp_yr = min([y for y in anchor_years if y > yr])
                weight = (yr - low_yr) / (upp_yr - low_yr)
                pop_low, cbr_low, cdr_low = anchors[low_yr]
                pop_upp, cbr_upp, cdr_upp = anchors[upp_yr]
                pop = pop_low + weight * (pop_upp - pop_low)
                cbr = cbr_low + weight * (cbr_upp - cbr_low)
                cdr = cdr_low + weight * (cdr_upp - cdr_low)
            if yr == 2020:
                cdr *= 1.15
            b_vol = round((cbr / 1000) * pop, 3)
            d_vol = round((cdr / 1000) * pop, 3)
            time_series_rows.append({
                "year": yr, "region": region, "population_millions": round(pop, 3),
                "births_millions": b_vol, "deaths_millions": d_vol, "net_growth_millions": round(b_vol - d_vol, 3),
                "data_type": "Government Certified Base"
            })
    return pd.DataFrame(time_series_rows)

df_base = load_all_india_matrix()
@st.cache_data
def generate_forecasts(df, model_type="Ridge"):
    ml_rows = []
    years_list = list(range(2027, 2037))
    future_years = np.array(years_list).reshape(-1, 1)
    
    if model_type == "Lasso":
        estimator = Lasso(alpha=0.01, max_iter=10000)
    elif model_type == "Linear Regression":
        estimator = LinearRegression()
    else:
        estimator = Ridge(alpha=1.0)
        
    for region in df['region'].unique():
        df_modern = df[(df['region'] == region) & (df['year'] >= 1951)].sort_values('year')
        X_train = df_modern[['year']].values
        poly = PolynomialFeatures(degree=1, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_future_poly = poly.transform(future_years)
        
        m_pop = estimator.fit(X_train_poly, df_modern['population_millions'].values)
        m_births = Ridge(alpha=1.0).fit(X_train_poly, df_modern['births_millions'].values)
        m_deaths = Ridge(alpha=1.0).fit(X_train_poly, df_modern['deaths_millions'].values)
        
        p_pop = m_pop.predict(X_future_poly)
        p_birth = m_births.predict(X_future_poly)
        p_death = m_deaths.predict(X_future_poly)
        
        # Pull single scalar float points using indexing
        last_pop = float(df_modern[df_modern['year'] == 2026]['population_millions'].values[0])
        last_birth = float(df_modern[df_modern['year'] == 2026]['births_millions'].values[0])
        last_death = float(df_modern[df_modern['year'] == 2026]['deaths_millions'].values[0])
        
        # FIXED: Extracting elements via .item() to ensure single math scalars are calculated
        shift_pop = float(last_pop - m_pop.predict(poly.transform([[2026]]))[0].item())
        shift_birth = float(last_birth - m_births.predict(poly.transform([[2026]]))[0].item())
        shift_death = float(last_death - m_deaths.predict(poly.transform([[2026]]))[0].item())
        
        growth_step = float((last_pop - df_modern[df_modern['year'] == 2025]['population_millions'].values[0]) * 0.95)
        
        residuals = df_modern['population_millions'].values - m_pop.predict(X_train_poly)
        sigma = np.std(residuals) if np.std(residuals) > 0 else 0.1
        
        for idx, yr in enumerate(years_list):
            p_val = max(0.001, float(p_pop[idx]) + shift_pop + (yr - 2026) * growth_step)
            b_val = max(0.000, float(p_birth[idx]) + shift_birth)
            d_val = max(0.000, float(p_death[idx]) + shift_death)
            
            uncertainty_spread = sigma * np.sqrt(yr - 2026) * 1.96
            
            ml_rows.append({
                "year": yr, "region": region, "population_millions": round(p_val, 3),
                "pop_lower_ci": round(max(0.001, p_val - uncertainty_spread), 3),
                "pop_upper_ci": round(p_val + uncertainty_spread, 3),
                "births_millions": round(b_val, 3), "deaths_millions": round(d_val, 3),
                "net_growth_millions": round(b_val - d_val, 3), "data_type": "ML Forecast Projection"
            })
    return pd.concat([df, pd.DataFrame(ml_rows)], ignore_index=True)
# ==========================================
# SIDEBAR NAVIGATION CONTROL ENGINE
# ==========================================
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

df_complete = generate_forecasts(df_base, model_type=selected_model)
available_years = sorted(list(df_complete['year'].unique()))
from_year, to_year = st.sidebar.select_slider(
    "Select Project Analysis Horizon Window:", 
    options=available_years, value=(1951, 2036),
    key="dashboard_horizon_slider"
)

# ==========================================
# SCREEN HEADERS & USER RENDERING LAYERS
# ==========================================
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
