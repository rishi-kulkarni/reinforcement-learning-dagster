from pathlib import Path
from dagster import ConfigurableResource
from duckdb import connect
import joblib
from pandas import DataFrame
from bayesianbandits import (
    Arm,
    Bandit,
    NormalInverseGammaRegressor,
    contextual,
    thompson_sampling,
)


class DuckDBConnection(ConfigurableResource):
    filepath: str

    def query(self, query: str) -> DataFrame:
        conn = connect(self.filepath)
        res = conn.execute(query).df()
        conn.close()

        return res


def reward_func(x):
    return x


@contextual
class WallflowerBonusBandit(
    Bandit,
    learner=NormalInverseGammaRegressor(),
    policy=thompson_sampling(),
    delayed_reward=True,
):
    arm_0_coupon = Arm(0, reward_function=reward_func)
    arm_5_coupon = Arm(5, reward_function=reward_func)
    arm_10_coupon = Arm(10, reward_function=reward_func)
    arm_15_coupon = Arm(15, reward_function=reward_func)


class WallflowerBanditLoader(ConfigurableResource):
    filepath: str

    def load(self):
        if Path(self.filepath).exists():
            return joblib.load(self.filepath)
        else:
            model = WallflowerBonusBandit()
            joblib.dump(model, self.filepath)
            return model

    def full_refresh(self):
        model = WallflowerBonusBandit()
        joblib.dump(model, self.filepath)
        return model

    def save(self, model):
        joblib.dump(model, self.filepath)
