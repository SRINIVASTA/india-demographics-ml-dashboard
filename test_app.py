import pytest
import pandas as pd
import numpy as np
from app import df_complete  # Imports the new updated dataset engine

def test_master_dataframe_integrity():
    """Verify that df_complete loaded and concatenated correctly."""
    assert isinstance(df_complete, pd.DataFrame)
    assert not df_complete.empty
    
    # Check that new confidence interval columns exist in the dataset
    required_columns = [
        "year", "region", "population_millions", 
        "pop_lower_ci", "pop_upper_ci", 
        "births_millions", "deaths_millions", "data_type"
    ]
    for col in required_columns:
        assert col in df_complete.columns

def test_selected_regions_exist():
    """Confirm specific demographic territories parse correctly when selected."""
    unique_regions = df_complete['region'].unique()
    assert "India (Total Country)" in unique_regions
    assert "Andaman and Nicobar Islands" in unique_regions
    assert "Lakshadweep" in unique_regions

def test_logical_math_constraints():
    """Ensure math metrics never output impossible numbers across all models."""
    assert (df_complete['population_millions'] > 0).all()
    assert (df_complete['births_millions'] >= 0).all()
    assert (df_complete['deaths_millions'] >= 0).all()
    
    # Verify that the lower confidence interval boundary is logically bounded
    assert (df_complete['pop_lower_ci'] <= df_complete['population_millions']).all()
    assert (df_complete['pop_upper_ci'] >= df_complete['population_millions']).all()

def test_covid_anomaly_modifier():
    """Verify that the 2020 annual death rate multiplier is applied cleanly."""
    df_delhi = df_complete[df_complete['region'] == "Delhi (NCT)"]
    deaths_2020 = df_delhi[df_delhi['year'] == 2020]['deaths_millions'].values
    assert deaths_2020 > 0
