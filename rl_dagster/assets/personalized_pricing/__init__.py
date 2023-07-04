import hashlib
import warnings
from pathlib import Path
from typing import Tuple, cast

import joblib
import numpy as np
import pandas as pd
from bayesianbandits import (
    Arm,
    Bandit,
    NormalInverseGammaRegressor,
    contextual,
    thompson_sampling,
)
from dagster import (
    AssetSelection,
    AutoMaterializePolicy,
    Config,
    DataVersion,
    DefaultSensorStatus,
    DynamicPartitionsDefinition,
    ExperimentalWarning,
    Nothing,
    Output,
    RunRequest,
    SensorEvaluationContext,
    SensorResult,
    SkipReason,
    asset,
    define_asset_job,
    observable_source_asset,
    sensor,
)

from dagster._core.definitions.data_version import (
    extract_data_version_from_entry,
)
from dagster_duckdb_pandas import DuckDBPandasIOManager
from formulaic import ModelMatrix, ModelSpec, model_matrix

from ._resources import DuckDBConnection

warnings.filterwarnings("ignore", category=ExperimentalWarning)

DATA_DB = Path(__file__).parent / "../../../data/data.db"
MODEL = Path(__file__).parent / "../../../data/wallflower_bonus_bandit.pkl"

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
def check_for_new_data(pricing_conn: DuckDBConnection) -> DataVersion:
    df = pricing_conn.query("SELECT * FROM today_date")
    hashed = hashlib.sha256(df.to_csv().encode("utf-8")).hexdigest()

    return DataVersion(hashed)


daily_personalized_pricing_data_job = define_asset_job(
    "daily_personalized_pricing_data_job",
    AssetSelection.keys("daily_eligible_ips_design_matrix", "daily_bonus_success_data"),
    partitions_def=personalized_pricing_partitions_def,
)


@sensor(
    job=daily_personalized_pricing_data_job, default_status=DefaultSensorStatus.RUNNING
)
def daily_personalized_pricing_data_sensor(context: SensorEvaluationContext):
    cursor = context.cursor if context.cursor else None
    asset_event = context.instance.get_latest_data_version_record(
        check_for_new_data.key
    )
    if asset_event is None:
        return SkipReason(skip_message="No observations yet.")
    data_version = extract_data_version_from_entry(asset_event.event_log_entry).value

    if cursor is None or data_version != cursor:
        context.log.info("New data detected, advancing cursor")

        current_date = pd.Timestamp.now().floor("S").strftime("%Y-%m-%d %H:%M:%S")

        return SensorResult(
            [
                RunRequest(
                    run_key=f"daily_personalized_pricing_data_{data_version}",
                    partition_key=current_date,
                )
            ],
            dynamic_partitions_requests=[
                personalized_pricing_partitions_def.build_add_request([current_date])
            ],
            cursor=data_version,
        )


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


def reward_func(x):
    return x


@contextual
class WallflowerBonusBandit(
    Bandit,
    learner=NormalInverseGammaRegressor(),
    policy=thompson_sampling(),
    delayed_reward=True,
):
    arm_0_bonus = Arm(0, reward_function=reward_func)
    arm_2_bonus = Arm(2, reward_function=reward_func)
    arm_3_bonus = Arm(3, reward_function=reward_func)
    arm_4_bonus = Arm(4, reward_function=reward_func)


class BonusAmountBanditConfig(Config):
    full_refresh: bool = False


@asset(
    partitions_def=personalized_pricing_partitions_def,
    auto_materialize_policy=AutoMaterializePolicy.eager(),
)
def bonus_amount_bandit(
    context,
    config: BonusAmountBanditConfig,
    daily_bonus_success_data: Tuple[pd.Series, pd.Series, np.ndarray, pd.Series],
) -> Output[None]:
    if MODEL.exists() and not config.full_refresh:
        context.log.info("Loading existing model")
        agent = joblib.load(MODEL)
    else:
        context.log.info("Creating new model")
        agent = WallflowerBonusBandit()

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
    joblib.dump(agent, MODEL)

    context.log.info("Done")

    return Output(Nothing(), metadata=metadata)


@asset(
    io_manager_def=DuckDBPandasIOManager(database=str(DATA_DB)),
    partitions_def=personalized_pricing_partitions_def,
    metadata={"partition_expr": "start_date"},
    non_argument_deps={"bonus_amount_bandit"},
    auto_materialize_policy=AutoMaterializePolicy.eager(),
)
def bonus_description(
    daily_eligible_ips_design_matrix: Tuple[pd.Series, pd.Series, np.ndarray],
) -> Output[pd.DataFrame]:
    user_ids, dates_eligible, design_matrix = daily_eligible_ips_design_matrix

    unique_ids = [
        f"{user_id}_{date}" for user_id, date in zip(user_ids, dates_eligible)
    ]

    agent = joblib.load(MODEL)

    assignments = [
        agent.pull(row, unique_id=unique_id)
        for row, unique_id in zip(design_matrix, unique_ids)
    ]

    joblib.dump(agent, MODEL)

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
