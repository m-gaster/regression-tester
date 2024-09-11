import warnings
from pathlib import Path

import polars as pl


class ColumnsNotEqualError(Exception):
    pass


class LengthsNotEqualError(Exception):
    pass


class OverlappingContentsUneqalError(Exception):
    pass


class SchemaDifferenceError(Exception):
    pass


def compare_schemata(
    name1: str,
    name2: str,
    schema1: pl.Schema | dict[str, pl.DataType],
    schema2: pl.Schema | dict[str, pl.DataType],
    raise_if_schema_difference: bool = True,
) -> None:
    schema_differences = {}
    for col in schema1.keys():
        type1 = schema1[col]
        type2 = schema2[col]
        if type1 != type2:
            schema_differences[col] = (type1, type2)

    if schema_differences:
        if raise_if_schema_difference:
            raise SchemaDifferenceError(
                f"Schema difference between {name1} and {name2}: {schema_differences}"
            )
        else:
            warnings.warn(
                f"Schema difference between {name1} and {name2}: {schema_differences}",
                UserWarning,
            )


def compare_dataframes(
    df1: pl.DataFrame,
    df2: pl.DataFrame,
    name1: str,
    name2: str,
    comparison_export_path: Path,
    raise_if_schema_difference: bool = True,
) -> None:
    in_df1_not_in_df2: list[str] = sorted(
        list(set(df1.columns).difference(set(df2.columns)))
    )
    in_df2_not_in_df1: list[str] = sorted(
        list(set(df2.columns).difference(set(df1.columns)))
    )
    if in_df1_not_in_df2 or in_df2_not_in_df1:
        raise ColumnsNotEqualError(
            f"In {name1} not in {name2}: {in_df1_not_in_df2}\
                In {name2} not in {name1}: {in_df2_not_in_df1}"
        )
    if df1.shape[0] != df2.shape[0]:
        raise LengthsNotEqualError(
            f"{name1} shape = {df1.shape}\
                {name2} shape = {df2.shape}"
        )

    # COMPARE SCHEMA
    compare_schemata(name1=name1, name2=name2, schema1=df1.schema, schema2=df2.schema)

    comparison = df1.to_pandas().compare(
        df2.to_pandas(), result_names=(name1, name2)
    )  # convert to pandas because it has a better comparison
    if not comparison.empty:
        unequal_cols: list[str] = []
        for col in df1.columns:
            if not df1[col].equals(df2[col]):
                unequal_cols.append(col)
        comparison.to_csv(comparison_export_path)
        raise OverlappingContentsUneqalError(f"{unequal_cols=}")
