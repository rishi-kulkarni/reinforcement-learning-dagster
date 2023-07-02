from dagster import Definitions, load_assets_from_modules

from .assets import personalized_pricing

assets = load_assets_from_modules(
    [personalized_pricing], group_name="personalized_pricing"
)

print(assets)

defs = Definitions(
    assets=assets, sensors=[personalized_pricing.bonus_description_sensor]
)
