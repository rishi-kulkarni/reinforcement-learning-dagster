import warnings
from hashlib import sha256
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
    AssetKey,
    AssetSelection,
    AutoMaterializePolicy,
    Config,
    DataVersion,
    DynamicPartitionsDefinition,
    ExperimentalWarning,
    MultiAssetSensorEvaluationContext,
    Nothing,
    Output,
    RunRequest,
    SensorResult,
    asset,
    define_asset_job,
    multi_asset_sensor,
    observable_source_asset,
    DefaultSensorStatus,
)
from dagster_duckdb_pandas import DuckDBPandasIOManager
from duckdb import connect
from formulaic import ModelMatrix, ModelSpec, model_matrix

warnings.filterwarnings("ignore", category=ExperimentalWarning)

DATA_DB = Path(__file__).parent / "../../../data/data.db"
MODEL = Path(__file__).parent / "../../../data/wallflower_bonus_bandit.pkl"


@asset
def context_model_spec() -> Output[ModelSpec]:
    """Generates a design matrix for the features we want to use in our model."""

    conn = connect(str(DATA_DB))

    df = conn.execute("SELECT distinct market, ptype FROM users").df()
    df.columns = df.columns.str.lower()

    formula = "market * ptype"

    ms = cast(ModelSpec, model_matrix(formula, df).model_spec)

    metadata = {
        "formula": formula,
        "columns_out": len(ms.column_names),
    }

    return Output(ms, metadata=metadata)


@observable_source_asset
def daily_eligible_ips_source_asset():
    conn = connect(str(DATA_DB))

    df = conn.execute("SELECT * FROM users_eligible_day").df()

    hash_sig = sha256(df.to_csv().encode()).hexdigest()

    return DataVersion(hash_sig)


@observable_source_asset
def daily_bonus_success_source_asset():
    conn = connect(str(DATA_DB))

    df = conn.execute("SELECT * FROM bonus_description_today").df()

    hash_sig = sha256(df.to_csv().encode()).hexdigest()

    return DataVersion(hash_sig)


@asset(
    non_argument_deps={"daily_eligible_ips_source_asset"},
    auto_materialize_policy=AutoMaterializePolicy.eager(),
)
def daily_eligible_ips_design_matrix(
    context_model_spec: ModelSpec,
) -> Output[Tuple[pd.Series, pd.Series, np.ndarray]]:
    """Generates a design matrix for the features we want to use in our model."""

    conn = connect(str(DATA_DB))

    df = conn.execute("SELECT * FROM users_eligible_day").df()
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
    non_argument_deps={"daily_bonus_success_source_asset"},
    auto_materialize_policy=AutoMaterializePolicy.eager(),
)
def daily_bonus_success_data(
    context_model_spec: ModelSpec,
) -> Output[Tuple[pd.Series, pd.Series, np.ndarray, pd.Series]]:
    """Generates a design matrix for the features we want to use in our model."""

    conn = connect(str(DATA_DB))

    df = conn.execute(
        """select bonus_description_today.user_id, start_date, market, ptype,
           bonus_amount
           from bonus_description_today
           join users on bonus_description_today.user_id = users.user_id
           and bonus_description_today.start_date = users.date_eligible
           """
    ).df()
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


bonus_description_partitions_def = DynamicPartitionsDefinition(name="bonus_description")


@asset(
    io_manager_def=DuckDBPandasIOManager(database=str(DATA_DB)),
    partitions_def=bonus_description_partitions_def,
    metadata={"partition_expr": "start_date"},
    non_argument_deps={"bonus_amount_bandit"},
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


bonus_description_job = define_asset_job(
    "bonus_description_job",
    AssetSelection.keys("bonus_description"),
    partitions_def=bonus_description_partitions_def,
)


@multi_asset_sensor(
    monitored_assets=[
        AssetKey("daily_eligible_ips_design_matrix"),
        AssetKey("bonus_amount_bandit"),
    ],
    job=bonus_description_job,
    default_status=DefaultSensorStatus.RUNNING,
)
def bonus_description_sensor(context: MultiAssetSensorEvaluationContext):
    asset_events = context.latest_materialization_records_by_key().values()

    conn = connect(str(DATA_DB))
    df = conn.execute("SELECT * FROM today_date").df()

    key = df.iloc[0][0].date()

    if any(asset_events):
        context.advance_all_cursors()

        return SensorResult(
            [
                RunRequest(
                    run_key=f"bonus_description_{key.strftime('%Y-%m-%d')}",
                    partition_key=key.strftime("%Y-%m-%d"),
                )
            ],
            dynamic_partitions_requests=[
                bonus_description_partitions_def.build_add_request(
                    [key.strftime("%Y-%m-%d")]
                )
            ],
        )
