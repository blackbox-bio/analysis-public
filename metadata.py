from palmreader_analysis.variants import ColumnMetadata
from cols_name_dicts import summary_col_name_dict


def _update_metadata(column: ColumnMetadata) -> ColumnMetadata:
    column["column"] = summary_col_name_dict[column["column"]]

    return column


if __name__ == "__main__":
    import json
    from palmreader_analysis import SummaryContext, summary
    from itertools import chain

    columns = SummaryContext.get_all_columns()

    columns = list(chain.from_iterable(map(lambda col: col.metadata(), columns)))

    # apply name dictionary
    columns = list(map(_update_metadata, columns))

    print(json.dumps(columns, default=lambda enum: enum.value), end="")
