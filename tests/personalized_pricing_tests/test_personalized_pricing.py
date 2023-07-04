from typing import Optional
import numpy as np
import pandas as pd
from pandas import DataFrame
from pydantic import PrivateAttr
from rl_dagster.assets.personalized_pricing import (
    DuckDBConnection,
    WallflowerBanditLoader,
)
from rl_dagster.assets.personalized_pricing._resources import WallflowerBonusBandit
from rl_dagster.assets.personalized_pricing._assets import (
    context_model_spec,
    daily_bonus_success_data,
    daily_eligible_ips_design_matrix,
    bonus_amount_bandit,
    bonus_description,
)

from dagster import materialize_to_memory, DagsterInstance


class MockDuckDBConnection(DuckDBConnection):
    def query(self, query: str) -> DataFrame:
        match query:
            case "SELECT distinct market, ptype FROM users":
                return DataFrame(
                    {
                        "MARKET": ["BOS-WEST", "BOS-WEST", "BOS-EAST", "BOS-EAST"],
                        "PTYPE": ["CNA", "LPN", "CNA", "LPN"],
                    }
                )
            case "SELECT * FROM today_date":
                return DataFrame({"today": ["2021-01-01"]})
            case "SELECT * FROM users_eligible_day":
                return DataFrame(
                    {
                        "USER_ID": [1, 2, 3],
                        "MARKET": ["BOS-WEST", "BOS-WEST", "BOS-EAST"],
                        "PTYPE": ["CNA", "LPN", "CNA"],
                        "DATE_ELIGIBLE": pd.to_datetime(
                            ["2021-01-01", "2021-01-01", "2021-01-01"]
                        ),
                        "TODAY": pd.to_datetime(
                            ["2021-01-01", "2021-01-01", "2021-01-01"]
                        ),
                    }
                )
            case """select bonus_description_today.user_id, start_date, market, ptype,
           bonus_amount
           from bonus_description_today
           join users on bonus_description_today.user_id = users.user_id
           and bonus_description_today.start_date = users.date_eligible
           """:
                return DataFrame(
                    {
                        k: pd.Series(dtype=t)
                        for k, t in zip(
                            [
                                "USER_ID",
                                "START_DATE",
                                "MARKET",
                                "PTYPE",
                                "BONUS_AMOUNT",
                            ],
                            [np.int64, "datetime64[ns]", str, str, np.int64],
                        )
                    }
                )
            case _:
                raise NotImplementedError


class MockBanditLoader(WallflowerBanditLoader):
    _cached_model: Optional[WallflowerBonusBandit] = PrivateAttr(None)

    def load(self):
        if self._cached_model is None:
            self._cached_model = WallflowerBonusBandit()
        return self._cached_model

    def full_refresh(self):
        self._cached_model = WallflowerBonusBandit()
        return self._cached_model

    def save(self, model):
        self._cached_model = model


def test_integration():
    resources = {
        "pricing_conn": MockDuckDBConnection(filepath=""),
        "bandit_loader": MockBanditLoader(filepath=""),
    }

    assets = [
        context_model_spec,
        daily_eligible_ips_design_matrix,
        daily_bonus_success_data,
        bonus_amount_bandit,
        bonus_description,
    ]

    instance = DagsterInstance.ephemeral()
    instance.add_dynamic_partitions("personalized_pricing", ["2021-01-01"])

    result = materialize_to_memory(
        assets, resources=resources, partition_key="2021-01-01", instance=instance
    )

    assert result.success
