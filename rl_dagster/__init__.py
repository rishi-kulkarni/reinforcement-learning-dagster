from pathlib import Path
from dagster import Definitions
from dagster_duckdb_pandas import DuckDBPandasIOManager

from .assets.personalized_pricing import (
    assets,
    coupon_sensors,
    DuckDBConnection,
    WallflowerBanditLoader,
)


DATA_DB = Path(__file__).parent / "../data/data.db"
MODEL = Path(__file__).parent / "../data/wallflower_bonus_bandit.pkl"

defs = Definitions(
    assets=assets,
    sensors=coupon_sensors,
    resources={
        "pricing_conn": DuckDBConnection(filepath=str(DATA_DB)),
        "duck_db_io": DuckDBPandasIOManager(database=str(DATA_DB)),
        "bandit_loader": WallflowerBanditLoader(filepath=str(MODEL)),
    },
)
