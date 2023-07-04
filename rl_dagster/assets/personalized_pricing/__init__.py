from dagster import load_assets_from_modules

from . import _assets as pricing_assets
from ._resources import DuckDBConnection, WallflowerBanditLoader
from ._sensors import pricing_sensors

assets = load_assets_from_modules(
    [pricing_assets],
    group_name="personalized_pricing",
)

sensors = pricing_sensors
