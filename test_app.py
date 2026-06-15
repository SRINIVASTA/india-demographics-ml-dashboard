import pytest
import pandas as pd
import numpy as np
from app import df_master  # Automatically imports your master data table engine

def test_master_dataframe_integrity():
    """Verify that df_master loaded and concatenated correctly."""
    assert isinstance(df_master, pd.DataFrame)
    assert not df_master.empty
    
    # Check that structural target data fields are fully generated
    required_columns = ["year", "region", "population_millions", "births_millions", "deaths_millions", "data_type"]
    for col in required_columns:
        assert col in df_master.columns

def test_selected_regions_exist():
    """Confirm specific demographic territories parse correctly when selected."""
    unique_regions = df_master['region'].unique()
    
    # Check national country total baseline
    assert "India (Total Country)" in unique_regions
    
    # Check distinct island territory locations match your requested updates
    assert "Andaman and Nicobar Islands" in unique_regions
    assert "Lakshadweep" in unique_regions

def test_simulation_timeline_bounds():
    """Verify data spans the entire responsive slider range timeline (1901 to 2036)."""
    assert df_master['year'].min() == 1901
    assert df_master['year'].max() == 2036

def test_logical_math_constraints():
    """Ensure math metrics never output impossible numbers when a region is selected."""
    # Populations must stay positive and real across all future projections
    assert (df_master['population_millions'] > 0).all()
    
    # Live occurrences cannot be less than zero
    assert (df_master['births_millions'] >= 0).all()
    assert (df_master['deaths_millions'] >= 0).all()
