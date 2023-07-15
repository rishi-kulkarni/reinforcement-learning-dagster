import warnings
from typing import Tuple, cast

import numpy as np
import pandas as pd
from dagster import (
    AssetExecutionContext,
    AssetIn,
    Config,
    DynamicPartitionsDefinition,
    ExperimentalWarning,
    LastPartitionMapping,
    Nothing,
    Output,
    asset,
)

from formulaic import ModelMatrix, ModelSpec, model_matrix

from ._resources import DuckDBConnection, WallflowerBanditLoader
from ._simulate_margin import simulated_margin

warnings.filterwarnings("ignore", category=ExperimentalWarning)


send_coupons_partitions_def = DynamicPartitionsDefinition(
    name="send_coupons_partitions",
)
update_coupons_model_partitions_def = DynamicPartitionsDefinition(
    name="update_coupons_model_partitions",
)


@asset
def context_model_spec(pricing_conn: DuckDBConnection) -> Output[ModelSpec]:
    """Generates a design matrix for the features we want to use in our model."""

    df = pricing_conn.query("SELECT distinct geo, device FROM coupon_eligible_users;")
    df.columns = df.columns.str.lower()

    formula = "1 + geo + device"

    ms = cast(ModelSpec, model_matrix(formula, df, ensure_full_rank=False).model_spec)

    metadata = {
        "formula": formula,
        "columns_out": len(ms.column_names),
    }

    return Output(ms, metadata=metadata)


@asset(
    partitions_def=send_coupons_partitions_def,
)
def daily_eligible_ips_design_matrix(
    context_model_spec: ModelSpec,
    pricing_conn: DuckDBConnection,
) -> Output[Tuple[pd.Series, pd.Series, np.ndarray]]:
    """Generates a design matrix for the features we want to use in our model."""

    df = pricing_conn.query("SELECT * FROM users_eligible_today")
    df.columns = df.columns.str.lower()

    design_matrix: ModelMatrix[np.float_] = context_model_spec.get_model_matrix(df)
    metadata = {
        "total_obs": len(design_matrix),
    }

    return Output(
        (df.user_id, df.date_eligible.dt.date, np.asarray(design_matrix)),
        metadata=metadata,
    )


@asset(
    partitions_def=update_coupons_model_partitions_def,
)
def daily_model_update_data(
    context_model_spec: ModelSpec,
    pricing_conn: DuckDBConnection,
) -> Output[Tuple[pd.Series, pd.Series, np.ndarray, pd.Series]]:
    """Generates a design matrix for the features we want to use in our model."""

    df = pricing_conn.query(
        """select margins_to_check.user_id, start_date, geo, device,
           bonus_amount
           from margins_to_check
           join coupon_eligible_users using (user_id)
           """
    )
    df.columns = df.columns.str.lower()

    df["margin"] = [
        simulated_margin(row.geo, row.device, row.bonus_amount)
        for row in df.itertuples()
    ]

    design_matrix: ModelMatrix[np.float_] = context_model_spec.get_model_matrix(df)
    metadata = {
        "total_obs": len(design_matrix),
    }

    return Output(
        (df.user_id, df.start_date.dt.date, np.asarray(design_matrix), df.margin),
        metadata=metadata,
    )


class BonusAmountBanditConfig(Config):
    full_refresh: bool = False


@asset(
    ins={
        "daily_model_update_data": AssetIn(
            "daily_model_update_data", partition_mapping=LastPartitionMapping()
        )
    },
)
def bonus_amount_bandit(
    context: AssetExecutionContext,
    config: BonusAmountBanditConfig,
    bandit_loader: WallflowerBanditLoader,
    daily_model_update_data: Tuple[pd.Series, pd.Series, np.ndarray, pd.Series],
) -> Output[None]:
    if config.full_refresh:
        context.log.info("Creating new model")
        agent = bandit_loader.full_refresh()
    else:
        agent = bandit_loader.load()

    user_ids, dates_eligible, design_matrix, margins = daily_model_update_data

    metadata = {}

    if len(user_ids) > 0:
        context.log.info(f"Updating model with {len(user_ids)} observations")
        unique_ids = [
            f"{user_id}_{date}" for user_id, date in zip(user_ids, dates_eligible)
        ]
        agent.update(design_matrix, np.atleast_1d(margins), unique_id=unique_ids)

        metadata = {
            f"{name}_global_intercept": arm.learner.coef_[0]
            for name, arm in agent.arms.items()
        }

    context.log.info("Saving model")
    bandit_loader.save(agent)
    context.log.info("Done")

    return Output(Nothing(), metadata=metadata)


@asset(
    io_manager_key="duck_db_io",
    partitions_def=send_coupons_partitions_def,
    metadata={"partition_expr": "start_date"},
    non_argument_deps={"bonus_amount_bandit"},
)
def coupon_offers(
    daily_eligible_ips_design_matrix: Tuple[pd.Series, pd.Series, np.ndarray],
    bandit_loader: WallflowerBanditLoader,
) -> Output[pd.DataFrame]:
    user_ids, dates_eligible, design_matrix = daily_eligible_ips_design_matrix

    unique_ids = [
        f"{user_id}_{date}" for user_id, date in zip(user_ids, dates_eligible)
    ]

    agent = bandit_loader.load()

    assignments = [
        agent.pull(row, unique_id=unique_id)
        for row, unique_id in zip(design_matrix, unique_ids)
    ]

    bandit_loader.save(agent)

    df = pd.DataFrame(
        {
            "user_id": user_ids,
            "start_date": dates_eligible,
            "end_date": (dates_eligible + pd.Timedelta(days=3)),
            "bonus_amount": assignments,
        }
    )

    metadata = {
        "total_obs": len(df),
        "average_bonus": df.bonus_amount.mean(),
    }

    return Output(df, metadata=metadata)
