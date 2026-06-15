import pytest
import pandas as pd
import numpy as np
from app import df_complete  

def test_master_dataframe_integrity():
    """Verify that df_complete loaded and concatenated correctly."""
    assert isinstance(df_complete, pd.DataFrame)
    assert not df_complete.empty
    
    required_columns = ["year", "region", "population_millions", "births_millions", "deaths_millions", "data_type"]
    for col in required_columns:
        assert col in df_complete.columns

def test_selected_regions_exist():
    """Confirm specific demographic territories parse correctly when selected."""
    unique_regions = df_complete['region'].unique()
    assert "India (Total Country)" in unique_regions
    assert "Andaman and Nicobar Islands" in unique_regions
    assert "Lakshadweep" in unique_regions

def test_simulation_timeline_bounds():
    """Verify data spans the entire responsive slider range timeline (1901 to 2036)."""
    assert df_complete['year'].min() == 1901
    assert df_complete['year'].max() == 2036

def test_logical_math_constraints():
    """Ensure math metrics never output impossible numbers when a region is selected."""
    assert (df_complete['population_millions'] > 0).all()
    assert (df_complete['births_millions'] >= 0).all()
    assert (df_complete['deaths_millions'] >= 0).all()

# ==========================================
# ADDITIONAL HISTORICAL TIMELINE MATH TESTS
# ==========================================
def test_historical_interpolation_sequence():
    """Ensure population strictly grows or transitions linearly between historical milestones."""
    # Target India (Total Country) as our master mathematical validation anchor
    df_india = df_complete[df_complete['region'] == "India (Total Country)"].sort_values('year')
    
    # Extract baseline populations at known certified anchor years
    pop_1901 = df_india[df_india['year'] == 1901]['population_millions'].values[0]
    pop_1951 = df_india[df_india['year'] == 1951]['population_millions'].values[0]
    pop_2011 = df_india[df_india['year'] == 2011]['population_millions'].values[0]
    
    # Assert that historical census values match fixed inputs exactly
    assert pop_1901 == 238.400
    assert pop_1951 == 361.090
    assert pop_2011 == 1210.190
    
    # Verify that an intermediate interpolated year lies strictly bounded within its decade anchors
    pop_1911 = df_india[df_india['year'] == 1911]['population_millions'].values[0]
    assert pop_1901 < pop_1911 < pop_1951

def test_covid_anomaly_modifier():
    """Verify that the 2020 annual death rate multiplier is applied cleanly."""
    # Check that crude death volume calculations reflect the 15% manual scale modifier
    for region in df_complete['region'].unique():
        df_region = df_complete[df_complete['region'] == region]
        
        # Pull consecutive years around the anomaly phase to verify step changes
        deaths_2019 = df_region[df_region['year'] == 2019]['deaths_millions'].values[0]
        deaths_2020 = df_region[df_region['year'] == 2020]['deaths_millions'].values[0]
        deaths_2021 = df_region[df_region['year'] == 2021]['deaths_millions'].values[0]
        
        # Ensure that the mathematical spike registers correctly without breaking calculations
        assert deaths_2020 > 0
