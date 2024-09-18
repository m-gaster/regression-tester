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
        The root path should contain the following files (unless you specify them yourself):
        - raw.txt  |  raw.txt.gz
        - processed.parquet
    """

    root_path: Path
    # name: str | None
    extraction_fnc: Callable[[Path], pl.DataFrame]
    cols_to_exclude: list[str] = Field(default_factory=list)
    optional_raw_input_path: Path | None = None
    optional_locally_processed_path: Path | None = None
    optional_processed_path: Path | None = None
    raise_if_schema_difference: bool = True
    ################################################################

    @property
    def RAW_INPUT_PATH(self) -> Path:
        if self.optional_raw_input_path:
            if not self.optional_raw_input_path.exists():
                raise FileNotFoundError(self.optional_raw_input_path)
            return self.optional_raw_input_path
        for filename in ("raw.txt", "raw.txt.gz"):
            filepath = self.root_path / filename
            if filepath.exists():
                return filepath
        raise FileNotFoundError(list(self.root_path.glob("*")))

    @property
    def PROCESSED_PATH(self) -> Path:
        if self.optional_processed_path:
            return self.optional_processed_path
        path: Path = self.root_path / "processed.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    @property
    def comparison_export_path(self) -> Path:
        return Path(self.root_path / "reg_test_comparison.csv")

    ################################################################

    def _exclude_cols(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.pipe(_sort_cols)
        present_cols_to_drop = [
            col for col in self.cols_to_exclude if col in df.columns
        ]

        cols_to_exclude_that_are_not_present = set(self.cols_to_exclude) - set(
            present_cols_to_drop
        )
        if cols_to_exclude_that_are_not_present:
            warning_str = f"{cols_to_exclude_that_are_not_present=}"
            warnings.warn(warning_str)

        return df.drop(present_cols_to_drop)

    @property
    def locally_processed_df(self) -> pl.DataFrame:
        df = self.extraction_fnc(self.RAW_INPUT_PATH)
        return self._exclude_cols(df)

    @property
    def ground_truth(self) -> pl.DataFrame:
        df = pl.read_parquet(self.PROCESSED_PATH)
        return self._exclude_cols(df)

    ################################################################

    def execute_regression_test(self) -> None:
        compare_dataframes(
            self.locally_processed_df,
            self.ground_truth,
            name1="Locally Processed",
            name2="Ground Truth",
            comparison_export_path=self.comparison_export_path,
            raise_if_schema_difference=self.raise_if_schema_difference,
        )

    ################################################################

    def overwrite_snapshot_w_local(self) -> None:
        warnings.warn("OVERWRITING SNAPSHOT WITH LOCAL DATA. This cannot be undone!")
        if self.cols_to_exclude:
            excluded_cols_warning: str = (
                f"NOT INCLUDING {self.cols_to_exclude=} IN SNAPSHOT OVERWRITE."
            )
            warnings.warn(excluded_cols_warning)
        # confirm overwrite via command line
        response: Literal["O"] | str | None = input(
            f'Type "O" to overwrite snapshot at {self.PROCESSED_PATH}.\n'
        )
        match response:
            case "O":
                print("\nOverwriting snapshot...")
                self.locally_processed_df.write_parquet(self.PROCESSED_PATH)
                print(f"\nSnapshot overwritten at {self.PROCESSED_PATH}.")

            case _:
                print("\nSnapshot not overwritten.")
