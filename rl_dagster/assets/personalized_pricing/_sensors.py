from dagster import (
    AssetSelection,
    DefaultSensorStatus,
    RunRequest,
    SensorEvaluationContext,
    SensorResult,
    SkipReason,
    define_asset_job,
    sensor,
)
import pandas as pd

from ._assets import personalized_pricing_partitions_def, check_for_new_data

from dagster._core.definitions.data_version import (
    extract_data_version_from_entry,
)

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


pricing_sensors = [daily_personalized_pricing_data_sensor]
