import re
from pathlib import Path
from unittest import mock

import polars as pl
import pytest

from regression_tester import RegressionTestPackage
from regression_tester.compare_dfs import (
    ColumnsNotEqualError,
    LengthsNotEqualError,
    OverlappingContentsUneqalError,
    SchemaDifferenceError,
)

MEANINGLESS_PATH = Path("abcd")

MOCK_SNAPSHOT_DF: pl.DataFrame = pl.DataFrame(
    {
        "col1": [1, 2, 3],
        "col2": ["a", "b", "c"],
        "col3": [1.1, 2.2, 3.3],
    }
)

MOCK_SNAPSHOT_DF_COLS_UNEQUAL = MOCK_SNAPSHOT_DF.select(pl.col("col1"))

MOCK_SNAPSHOT_DF_COL1_PLUS1 = MOCK_SNAPSHOT_DF.with_columns(pl.col("col1").add(1))
MOCK_SNAPSHOT_DF_SCHEMA_CHANGE = MOCK_SNAPSHOT_DF.with_columns(
    pl.col("col1").cast(pl.Float32)
)

MOCK_SNAPSHOT_DF_LENGTHS_UNEQUAL = pl.concat(
    (
        MOCK_SNAPSHOT_DF,
        pl.DataFrame({"col1": [None], "col2": [None], "col3": [None]}),
    )
)


def generalized_regression_test(
    tmp_path: Path,
    df1: pl.DataFrame,
    df2: pl.DataFrame,
    expected_exception: type[Exception] | None = None,
    expected_exception_content: str | None = None,
):
    # Create the paths for the mock processed and locally processed parquet files
    processed_parquet_path = tmp_path / "processed.parquet"
    locally_processed_parquet_path = tmp_path / "raw.txt"  # Simulating raw.txt presence

    # Write df2 (the ground truth dataframe) to the processed.parquet file
    df2.write_parquet(processed_parquet_path)

    # Write df1 to simulate raw.txt (locally processed dataframe)
    df1.write_parquet(locally_processed_parquet_path)

    # Mock the extraction function to return df1 (the locally processed dataframe)
    mock_extraction_fnc = mock.Mock(
        return_value=pl.read_parquet(locally_processed_parquet_path)
    )

    # Create an instance of the RegressionTestPackage with the temp path and mock extraction function
    package = RegressionTestPackage(
        root_path=tmp_path, extraction_fnc=mock_extraction_fnc, cols_to_exclude=[]
    )

    # Execute the regression test
    if expected_exception is not None:
        with pytest.raises(
            expected_exception,
            match=re.escape(expected_exception_content),  # type: ignore
        ):
            package.execute_regression_test()
    else:
        package.execute_regression_test()

    # Assert that the extraction function was called once to retrieve the locally processed data
    mock_extraction_fnc.assert_called_once_with(locally_processed_parquet_path)


def test_generalized_regression_success(tmp_path: Path):
    generalized_regression_test(tmp_path, MOCK_SNAPSHOT_DF, MOCK_SNAPSHOT_DF)


def test_regression_failure_cols_unequal(tmp_path: Path):
    generalized_regression_test(
        tmp_path,
        MOCK_SNAPSHOT_DF,
        MOCK_SNAPSHOT_DF_COLS_UNEQUAL,
        expected_exception=ColumnsNotEqualError,
        expected_exception_content="In Locally Processed not in Ground Truth: ['col2', 'col3']                In Ground Truth not in Locally Processed: []",
    )


def test_regression_failure_col1_plus1(tmp_path: Path):
    generalized_regression_test(
        tmp_path,
        MOCK_SNAPSHOT_DF,
        MOCK_SNAPSHOT_DF_COL1_PLUS1,
        expected_exception=OverlappingContentsUneqalError,
        expected_exception_content="unequal_cols=['col1']",
    )


def test_regression_failure_schema_change(tmp_path: Path):
    generalized_regression_test(
        tmp_path,
        MOCK_SNAPSHOT_DF,
        MOCK_SNAPSHOT_DF_SCHEMA_CHANGE,
        expected_exception=SchemaDifferenceError,
        expected_exception_content="Schema difference between Locally Processed and Ground Truth: {'col1': (Int64, Float32)}",
    )


def test_regression_failure_lengths_unequal(tmp_path: Path):
    generalized_regression_test(
        tmp_path,
        MOCK_SNAPSHOT_DF,
        MOCK_SNAPSHOT_DF_LENGTHS_UNEQUAL,
        expected_exception=LengthsNotEqualError,
        expected_exception_content="Locally Processed shape = (3, 3)                Ground Truth shape = (4, 3)",
    )
