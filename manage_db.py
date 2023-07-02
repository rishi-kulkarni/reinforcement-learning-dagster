import argparse
from pathlib import Path

import duckdb

DATA_DB = Path(__file__).parent / "./data/data.db"
MODEL = Path(__file__).parent / "./data/wallflower_bonus_bandit.pkl"


def reset():
    conn = duckdb.connect(str(DATA_DB))

    conn.execute("DELETE FROM public.bonus_description")
    conn.execute("UPDATE today_date SET today = '2023-07-02'")

    Path(MODEL).unlink(missing_ok=True)


def increment():
    conn = duckdb.connect(str(DATA_DB))

    conn.execute("UPDATE today_date SET today = today + INTERVAL 1 DAY")


def setdate(date_str):
    conn = duckdb.connect(str(DATA_DB))

    conn.execute(f"UPDATE today_date SET today = '{date_str}'")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()

    # Add reset subcommand
    parser.add_argument(
        "subcommand",
        choices=["reset", "increment", "setdate"],
        help="Subcommand to run",
    )

    # Set optional date argument
    parser.add_argument(
        "--date",
        help="Date to set",
    )

    args = parser.parse_args()

    if args.subcommand == "reset":
        reset()
    elif args.subcommand == "increment":
        increment()
    elif args.subcommand == "setdate":
        setdate(args.date)
    else:
        raise ValueError(f"Unknown subcommand {args.subcommand}")

    print("Done!")
