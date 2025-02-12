if __name__ == "__main__":
    import json
    from palmreader_analysis import SummaryContext, summary

    columns = SummaryContext.get_all_columns()

    columns = list(map(lambda col: col.metadata(), columns))

    print(json.dumps(columns, default=lambda enum: enum.value), end="")
