from dagster import asset

from duckdb import connect

from formulaic import ModelSpec, model_matrix


@asset
def context_model_spec():
    conn = connect(__file__ + "/../../data.db")

    df = conn.execute("SELECT distinct market, ptype FROM users").df()
