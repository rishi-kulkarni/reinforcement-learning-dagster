from dagster import Definitions, load_assets_from_modules

from rl_dagster.assets import personalized_pricing

assets = load_assets_from_modules([personalized_pricing])

defs = Definitions(assets=assets)
