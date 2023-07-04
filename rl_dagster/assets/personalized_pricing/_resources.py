from dagster import ConfigurableResource
from duckdb import connect
from pandas import DataFrame


class DuckDBConnection(ConfigurableResource):
    filepath: str

    def query(self, query: str) -> DataFrame:
        conn = connect(self.filepath)
        res = conn.execute(query).df()
        conn.close()

        return res
