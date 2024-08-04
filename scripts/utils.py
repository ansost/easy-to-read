"""
Script for utility functions
"""
import pandas as pd

def remove_zero_statements(filename):
    data = pd.read_csv(filename)

    data = data[data["num_statements"] !=0]

    return data
