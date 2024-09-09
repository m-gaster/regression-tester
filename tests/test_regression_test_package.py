from pathlib import Path

import polars as pl
import pytest
from regression_tester import RegressionTestPackage

# Mock extraction function
def mock_extraction_fnc(path: Path) -> pl.DataFrame:
    return pl.DataFrame(
        {"col1": [1, 2, 3], "col2": ["a", "b", "c"], "col3": [1.1, 2.2, 3.3]}
    )
mock_extraction_fnc(Path("")).write_parquet("test_data/processed.parquet")


@pytest.fixture
def temp_test_dir():
    # Create a temporary directory structure

    
    return Path("test_data")
    test_dir.mkdir()
    (test_dir / "raw.txt").write_text("mock raw data")

    # Create a mock processed.parquet file
    mock_df = mock_extraction_fnc(test_dir / "raw.txt")
    mock_df.write_parquet(test_dir / "processed.parquet")

    return test_dir


def test_regression_test_package_initialization(temp_test_dir):
    package = RegressionTestPackage(
        root_path=temp_test_dir, extraction_fnc=mock_extraction_fnc
    )

    assert package.root_path == temp_test_dir
    assert package.extraction_fnc == mock_extraction_fnc
    assert package.cols_to_exclude == []


def test_regression_test_package_properties(temp_test_dir):
    package = RegressionTestPackage(
        root_path=temp_test_dir, extraction_fnc=mock_extraction_fnc
    )

    assert package.RAW_INPUT_PATH == temp_test_dir / "raw.txt"
    assert package.PROCESSED_PATH == temp_test_dir / "processed.parquet"
    assert package.LOCALLY_PROCESSED_PATH == temp_test_dir / "locally_processed.parquet"
    assert package.comparison_export_path == temp_test_dir / "reg_test_comparison.csv"


def test_regression_test_package_dataframes(temp_test_dir): # FAILING
    package = RegressionTestPackage(
        root_path=temp_test_dir, extraction_fnc=mock_extraction_fnc
    )
    print(f"{package=}")

    locally_processed = package.locally_processed_df
    ground_truth = package.ground_truth

    assert isinstance(locally_processed, pl.DataFrame), locally_processed
    assert isinstance(ground_truth, pl.DataFrame)
    assert locally_processed.shape == ground_truth.shape
    assert locally_processed.columns == ground_truth.columns


def test_regression_test_package_execute(temp_test_dir, mocker): # FAILING
    package = RegressionTestPackage(
        root_path=temp_test_dir, extraction_fnc=mock_extraction_fnc
    )

    # Mock the compare_dataframes function
    mock_compare = mocker.patch(
        "src.pl_regression_tester._regression_tester.compare_dataframes"
    )

    package.execute_regression_test()

    # Assert that compare_dataframes was called with the correct arguments
    mock_compare.assert_called_once_with(
        package.locally_processed_df,
        package.ground_truth,
        name1="Locally Processed",
        name2="Ground Truth",
        comparison_export_path=package.comparison_export_path,
    )


def test_regression_test_package_with_excluded_cols(temp_test_dir):  # FAILING
    package = RegressionTestPackage(
        root_path=temp_test_dir,
        extraction_fnc=mock_extraction_fnc,
        cols_to_exclude=["col2"],
    )

    locally_processed = package.locally_processed_df
    ground_truth = package.ground_truth

    assert "col2" not in locally_processed.columns
    assert "col2" not in ground_truth.columns
    assert locally_processed.columns == ground_truth.columns == ["col1", "col3"]
