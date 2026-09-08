"""Turn an M x N frame into M x X of per-row value proportions."""
import numpy as np
import pandas as pd


def value_proportions(df, dropna=True, sort_columns=True, cumulative=False):
    """M x N -> M x X, where X is the unique values across the whole frame.

    out.loc[m, x] = (number of the N cells in row m equal to x) / N

    Each row sums to 1.0 when dropna=False, or to the row's non-null share
    when dropna=True (the denominator stays N either way, so a row with
    missing cells sums to < 1).

    dropna=True omits NaN from the output columns; False gives it a column.

    cumulative=True running-sums each row left to right, so out.loc[m, x] is
    the share of row m's N cells at or below x -- an empirical CDF across the
    value axis. That reading only holds if the columns are ordered, so it
    requires sort_columns=True, and dropna=True since a NaN column has no
    position in that order.
    """
    arr = df.to_numpy()
    n = arr.shape[1]

    vals = pd.unique(arr.ravel())
    isna = pd.isna(vals)
    out_vals = vals[~isna]
    if sort_columns:
        out_vals = np.sort(out_vals)

    # NaN != NaN, so equality counting handles nulls correctly by construction.
    data = {v: (arr == v).sum(axis=1) for v in out_vals}
    if not dropna and isna.any():
        data[np.nan] = pd.isna(arr).sum(axis=1)

    out = pd.DataFrame(data, index=df.index).div(n)

    if cumulative:
        if not sort_columns:
            raise ValueError("cumulative=True requires sort_columns=True; a "
                             "running sum over unordered columns is meaningless")
        if not dropna:
            raise ValueError("cumulative=True requires dropna=True; the NaN "
                             "column has no position in the value order")
        out = out.cumsum(axis=1)

    return out


if __name__ == "__main__":
    df = pd.DataFrame(
        [["a", "b", "a", "a"],
         ["b", "b", "b", "c"],
         ["c", "a", np.nan, "c"],
         ["a", "a", "a", "a"]],
        index=["m1", "m2", "m3", "m4"],
    )
    print("input (M=4, N=4):", df, sep="\n", end="\n\n")

    out = value_proportions(df)
    print("output (M=4, X=3):", out, sep="\n", end="\n\n")
    print("row sums:", out.sum(axis=1).to_dict(), end="\n\n")

    cum = value_proportions(df, cumulative=True)
    print("cumulative=True:", cum, sep="\n", end="\n\n")
    print("matches out.cumsum(axis=1):",
          np.allclose(cum.values, out.cumsum(axis=1).values),
          "| last column == row total:",
          np.allclose(cum.iloc[:, -1], out.sum(axis=1)), end="\n\n")

    for kw in ({"cumulative": True, "sort_columns": False},
               {"cumulative": True, "dropna": False}):
        try:
            value_proportions(df, **kw)
        except ValueError as e:
            print(f"rejects {kw}: {e}")
    print()

    # cross-check against the obvious slow path
    slow = df.apply(lambda r: r.value_counts(), axis=1).fillna(0).div(df.shape[1])
    print("matches per-row value_counts:",
          np.allclose(out.values, slow[out.columns].values), end="\n\n")

    # numeric dtype + a bigger shape
    rng = np.random.default_rng(0)
    big = pd.DataFrame(rng.integers(0, 5, size=(1000, 30)))
    b = value_proportions(big)
    print(f"numeric: {big.shape} -> {b.shape}, "
          f"all rows sum to 1: {np.allclose(b.sum(axis=1), 1.0)}")
    bc = value_proportions(big, cumulative=True)
    print("numeric cumulative: monotone rows:",
          bool((bc.diff(axis=1).iloc[:, 1:] >= -1e-12).all().all()),
          "| ends at 1.0:", np.allclose(bc.iloc[:, -1], 1.0))
