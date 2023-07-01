from pathlib import Path
from hashlib import sha256

from dagster import DataVersion, Output, asset, observable_source_asset
from duckdb import connect
from formulaic import ModelSpec, model_matrix

data_db = Path(__file__).parent / "../../../data/data.db"


@asset
def context_model_spec() -> Output[ModelSpec]:
    """Generates a design matrix for the features we want to use in our model."""

    conn = connect(str(data_db))

    df = conn.execute("SELECT distinct market, ptype FROM users").df()
    df.columns = df.columns.str.lower()

    formula = "market * ptype"

    ms = model_matrix(formula, df).model_spec

    metadata = {
        "formula": formula,
        "columns_out": len(ms.column_names),
    }

    return Output(ms, metadata=metadata)


@observable_source_asset
def daily_eligible_ips_source_asset():
    conn = connect(str(data_db))

    df = conn.execute("SELECT * FROM users_eligible_day").df()

    hash_sig = sha256(df.to_csv().encode()).hexdigest()

    return DataVersion(hash_sig)
