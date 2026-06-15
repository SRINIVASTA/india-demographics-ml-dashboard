import pytest
import pandas as pd
from app import load_all_india_matrix, generate_forecasts

def test_load_all_india_matrix_structure():
    df = load_all_india_matrix()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    
    required_columns = ["year", "region", "population_millions", "data_type"]
    for col in required_columns:
        assert col in df.columns

def test_generate_forecasts_bounds():
    df_base = load_all_india_matrix()
    df_ml = generate_forecasts(df_base)
    
    assert isinstance(df_ml, pd.DataFrame)
    
    # Updated to verify the complete historical and forecast range
    assert df_ml['year'].min() == 1901
    assert df_ml['year'].max() == 2036
    
    # Verify that the sequence contains all years in the range
    expected_years = set(range(1901, 2037))
    assert expected_years.issubset(set(df_ml['year']))
    
    assert (df_ml['population_millions'] > 0).all()
