from pathlib import Path
from dagster import Definitions, load_assets_from_modules

from .assets import personalized_pricing

assets = load_assets_from_modules(
    [personalized_pricing], group_name="personalized_pricing"
)

DATA_DB = Path(__file__).parent / "../data/data.db"

defs = Definitions(
    assets=assets,
    sensors=[personalized_pricing.daily_personalized_pricing_data_sensor],
    resources={
        "pricing_conn": personalized_pricing.DuckDBConnection(filepath=str(DATA_DB))
    },
)
