from dagster import (
    AssetKey,
    AssetSelection,
    DefaultSensorStatus,
    RunRequest,
    SensorEvaluationContext,
    SensorResult,
    SkipReason,
    sensor,
    define_asset_job,
)
import pandas as pd

from ._assets import send_coupons_partitions_def, update_coupons_model_partitions_def
from ._resources import DuckDBConnection

coupon_update_job = define_asset_job(
    name="send_coupon_update",
    selection="daily_eligible_ips_design_matrix",
    partitions_def=send_coupons_partitions_def,
)


@sensor(
    asset_selection=AssetSelection.keys(
        "daily_eligible_ips_design_matrix",
        "coupon_offers",
        "context_model_spec",
    ),
    default_status=DefaultSensorStatus.RUNNING,
)
def eligible_users_sensor(
    context: SensorEvaluationContext, pricing_conn: DuckDBConnection
) -> SkipReason | SensorResult:
    cursor = pd.to_datetime(context.cursor) if context.cursor else None

    max_date: pd.Timestamp = pricing_conn.query(
        "SELECT max(date_eligible) as date FROM users_eligible_today"
    )["date"][0]

    if cursor is None or max_date > cursor:
        assets_to_materialize = [
            AssetKey("daily_eligible_ips_design_matrix"),
            AssetKey("coupon_offers"),
        ]
        context.log.info("New data detected, advancing cursor")

        max_date_str = max_date.strftime("%Y-%m-%d")

        if cursor is None:
            context.log.info("Never run before, initializing context model spec")
            assets_to_materialize += [
                AssetKey("context_model_spec"),
            ]

        return SensorResult(
            run_requests=[
                RunRequest(
                    run_key=f"New_coupons_{max_date_str}",
                    partition_key=max_date_str,
                    asset_selection=assets_to_materialize,
                )
            ],
            dynamic_partitions_requests=[
                send_coupons_partitions_def.build_add_request([max_date_str])
            ],
            cursor=max_date_str,
        )
    return SkipReason("No new data detected")


@sensor(
    asset_selection=AssetSelection.keys(
        "daily_model_update_data", "bonus_amount_bandit"
    ),
    default_status=DefaultSensorStatus.RUNNING,
)
def margins_to_update_sensor(
    context: SensorEvaluationContext, pricing_conn: DuckDBConnection
) -> SkipReason | SensorResult:
    cursor = pd.to_datetime(context.cursor) if context.cursor else None

    max_date: pd.Timestamp = pricing_conn.query(
        "SELECT max(start_date) as date FROM margins_to_check"
    )["date"][0]

    # If there is no data in the table and we've run before, skip this run
    if max_date is pd.NaT and cursor is not None:
        return SkipReason("No new data detected")
    # If we've never run before or there is new data, run
    if cursor is None or max_date > cursor:
        context.log.info("New data detected, advancing cursor")

        if cursor is not None:
            context.log.info("New data detected, advancing cursor")
            partition_date_str = max_date.strftime("%Y-%m-%d")
        else:
            context.log.info("No cursor detected, running for first time")
            partition_date_str = "2023-07-01"

        return SensorResult(
            run_requests=[
                RunRequest(
                    run_key=f"Update_model_{partition_date_str}",
                    partition_key=partition_date_str,
                )
            ],
            dynamic_partitions_requests=[
                update_coupons_model_partitions_def.build_add_request(
                    [partition_date_str]
                )
            ],
            cursor=partition_date_str,
        )
    return SkipReason("No new data detected")


coupon_sensors = [margins_to_update_sensor, eligible_users_sensor]
