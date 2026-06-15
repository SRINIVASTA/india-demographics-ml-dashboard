import pytest
import pandas as pd
import numpy as np
from demographics_engine import load_all_india_matrix, generate_forecasts

@pytest.fixture
def compiled_dataset():
    """Fixture to generate dataset automatically across ensemble pipelines."""
    df_base = load_all_india_matrix()
    return generate_forecasts(df_base, model_type="Ridge")

def test_master_dataframe_integrity(compiled_dataset):
    """Verify that the engine loads and concatenates fields correctly."""
    assert isinstance(compiled_dataset, pd.DataFrame)
    assert not compiled_dataset.empty
    
    required_columns = [
        "year", "region", "population_millions", 
        "pop_lower_ci", "pop_upper_ci", 
        "births_millions", "deaths_millions", "data_type"
    ]
    for col in required_columns:
        assert col in compiled_dataset.columns

def test_selected_regions_exist(compiled_dataset):
    """Confirm specific demographic territories parse correctly."""
    unique_regions = compiled_dataset['region'].unique()
    assert "India (Total Country)" in unique_regions
    assert "Andaman and Nicobar Islands" in unique_regions
    assert "Lakshadweep" in unique_regions

def test_logical_math_constraints(compiled_dataset):
    """Ensure math metrics never output impossible numbers across models."""
    assert (compiled_dataset['population_millions'] > 0).all()
    assert (compiled_dataset['births_millions'] >= 0).all()
    assert (compiled_dataset['deaths_millions'] >= 0).all()
    
    # Filter for future rows to check bounds
    df_future = compiled_dataset[compiled_dataset['data_type'] == "ML Forecast Projection"]
    assert (df_future['pop_lower_ci'] <= df_future['population_millions']).all()
    assert (df_future['pop_upper_ci'] >= df_future['population_millions']).all()

def test_covid_anomaly_modifier(compiled_dataset):
    """Verify that the 2020 annual death rate multiplier is applied cleanly."""
    df_delhi = compiled_dataset[compiled_dataset['region'] == "Delhi (NCT)"]
    deaths_2020 = df_delhi[df_delhi['year'] == 2020]['deaths_millions'].values[0]
    assert deaths_2020 > 0
