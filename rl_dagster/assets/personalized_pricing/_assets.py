import hashlib
import warnings
from typing import Tuple, cast

import numpy as np
import pandas as pd
from dagster import (
    AssetExecutionContext,
    AssetKey,
    AutoMaterializePolicy,
    Config,
    DataVersion,
    DynamicPartitionsDefinition,
    ExperimentalWarning,
    Nothing,
    Output,
    asset,
    observable_source_asset,
)
from dagster._core.definitions.data_version import (
    extract_data_version_from_entry,
)
from formulaic import ModelMatrix, ModelSpec, model_matrix

from ._resources import DuckDBConnection, WallflowerBanditLoader

warnings.filterwarnings("ignore", category=ExperimentalWarning)


personalized_pricing_partitions_def = DynamicPartitionsDefinition(
    name="personalized_pricing"
)


@asset
def context_model_spec(pricing_conn: DuckDBConnection) -> Output[ModelSpec]:
    """Generates a design matrix for the features we want to use in our model."""

    df = pricing_conn.query("SELECT distinct market, ptype FROM users")
    df.columns = df.columns.str.lower()

    formula = "market * ptype"

    ms = cast(ModelSpec, model_matrix(formula, df).model_spec)

    metadata = {
        "formula": formula,
        "columns_out": len(ms.column_names),
    }

    return Output(ms, metadata=metadata)


@observable_source_asset
def check_for_new_data(
    context: AssetExecutionContext, pricing_conn: DuckDBConnection
) -> DataVersion:
    df = pricing_conn.query("SELECT * FROM today_date")
    hashed = hashlib.sha256(df.to_csv().encode("utf-8")).hexdigest()

    last_asset_event = context.instance.get_latest_data_version_record(
        AssetKey("check_for_new_data")
    )

    if last_asset_event is not None:
        last_hashed = extract_data_version_from_entry(
            last_asset_event.event_log_entry
        ).value
        if last_hashed == hashed:
            return DataVersion(hashed)
    context.log.info("New data detected, creating new partition")

    current_date = pd.Timestamp.now().floor("S").strftime("%Y-%m-%d %H:%M:%S")
    context.instance.add_dynamic_partitions("personalized_pricing", [current_date])

    return DataVersion(hashed)


@asset(
    auto_materialize_policy=AutoMaterializePolicy.eager(),
    partitions_def=personalized_pricing_partitions_def,
    non_argument_deps={"check_for_new_data"},
)
def daily_eligible_ips_design_matrix(
    context_model_spec: ModelSpec,
    pricing_conn: DuckDBConnection,
) -> Output[Tuple[pd.Series, pd.Series, np.ndarray]]:
    """Generates a design matrix for the features we want to use in our model."""

    df = pricing_conn.query("SELECT * FROM users_eligible_day")
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
    auto_materialize_policy=AutoMaterializePolicy.eager(),
    partitions_def=personalized_pricing_partitions_def,
    non_argument_deps={"check_for_new_data"},
)
def daily_bonus_success_data(
    context_model_spec: ModelSpec,
    pricing_conn: DuckDBConnection,
) -> Output[Tuple[pd.Series, pd.Series, np.ndarray, pd.Series]]:
    """Generates a design matrix for the features we want to use in our model."""

    df = pricing_conn.query(
        """select bonus_description_today.user_id, start_date, market, ptype,
           bonus_amount
           from bonus_description_today
           join users on bonus_description_today.user_id = users.user_id
           and bonus_description_today.start_date = users.date_eligible
           """
    )
    df.columns = df.columns.str.lower()

    df["margin"] = [
        np.random.normal(loc=80, scale=10) - 8 * row.bonus_amount
        if row.bonus_amount >= 3
        else 0
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
    partitions_def=personalized_pricing_partitions_def,
    auto_materialize_policy=AutoMaterializePolicy.eager(),
)
def bonus_amount_bandit(
    context,
    config: BonusAmountBanditConfig,
    bandit_loader: WallflowerBanditLoader,
    daily_bonus_success_data: Tuple[pd.Series, pd.Series, np.ndarray, pd.Series],
) -> Output[None]:
    if config.full_refresh:
        context.log.info("Creating new model")
        agent = bandit_loader.full_refresh()
    else:
        agent = bandit_loader.load()

    user_ids, dates_eligible, design_matrix, margins = daily_bonus_success_data

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
    partitions_def=personalized_pricing_partitions_def,
    metadata={"partition_expr": "start_date"},
    non_argument_deps={"bonus_amount_bandit"},
    auto_materialize_policy=AutoMaterializePolicy.eager(),
)
def bonus_description(
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
