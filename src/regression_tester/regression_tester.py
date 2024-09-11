import warnings
from pathlib import Path
from typing import Callable, Literal

import polars as pl
from pydantic import BaseModel, Field

from .compare_dfs import compare_dataframes


def _sort_cols(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(sorted(df.columns))


class RegressionTestPackage(BaseModel):
    """
    A package for performing regression tests on locally processed data.

    Attributes:
        root_path: The root path of the regression test package.
        extraction_fnc: A function that extracts a dataframe from a raw input path.

    NOTE:
        The root path should contain the following files:
        - raw.txt  |  raw.txt.gz
        - processed.parquet
    """

    root_path: Path
    # name: str | None
    extraction_fnc: Callable[[Path], pl.DataFrame]
    cols_to_exclude: list[str] = Field(default_factory=list)

    ################################################################

    @property
    def RAW_INPUT_PATH(self) -> Path:
        for filename in ("raw.txt", "raw.txt.gz"):
            filepath = self.root_path / filename
            if filepath.exists():
                return filepath
        raise FileNotFoundError(list(self.root_path.glob("*")))

    @property
    def PROCESSED_PATH(self) -> Path:
        path: Path = self.root_path / "processed.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    @property
    def LOCALLY_PROCESSED_PATH(self) -> Path:
        return self.root_path / "locally_processed.parquet"

    @property
    def comparison_export_path(self) -> Path:
        return Path(self.root_path / "reg_test_comparison.csv")

    ################################################################

    @property
    def locally_processed_df(self) -> pl.DataFrame:
        return (
            self.extraction_fnc(self.RAW_INPUT_PATH)
            .pipe(_sort_cols)
            .drop(self.cols_to_exclude)
        )

    @property
    def ground_truth(self) -> pl.DataFrame:
        return (
            pl.read_parquet(self.PROCESSED_PATH)
            .pipe(_sort_cols)
            .drop(self.cols_to_exclude)
        )

    ################################################################

    def execute_regression_test(self) -> None:
        compare_dataframes(
            self.locally_processed_df,
            self.ground_truth,
            name1="Locally Processed",
            name2="Ground Truth",
            comparison_export_path=self.comparison_export_path,
        )

    ################################################################

    def overwrite_snapshot_w_local(self) -> None:
        warnings.warn("OVERWRITING SNAPSHOT WITH LOCAL DATA. This cannot be undone!")
        # confirm overwrite via command line
        response: Literal["O"] | str | None = input(
            'Type "O" to overwrite snapshot at {self.PROCESSED_PATH}.\n'
        )
        match response:
            case "O":
                print("\nOverwriting snapshot...")
                self.locally_processed_df.write_parquet(self.PROCESSED_PATH)
                print(f"\nSnapshot overwritten at {self.PROCESSED_PATH}.")

            case _:
                print("\nSnapshot not overwritten.")
