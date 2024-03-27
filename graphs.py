from typing import List
from enum import Enum
import seaborn as sns
import pandas as pd

class GraphType(Enum):
    KDE = "kde"
    HIST = "hist"
    REG = "reg"

    def __str__(self):
        return self.value

def get_plot_fn(kind: str):
    _kind = GraphType(kind)

    if _kind == GraphType.KDE:
        return sns.kdeplot
    elif _kind == GraphType.HIST:
        return sns.histplot
    elif _kind == GraphType.REG:
        return sns.regplot
    else:
        raise ValueError(f"Unknown graph type: {_kind}")

def generate_pairgrid(
    summary_path: str,
    vars: List[str],
    hue: str,
    diag_kind: str,
    upper_kind: str,
    lower_kind: str,
    dest_path: str,
):
    # read summary features
    df = pd.read_csv(summary_path)

    # create pairgrid
    pg = sns.PairGrid(df, vars=vars, hue=hue)

    # map plots to pairgrid
    diag_plot = get_plot_fn(diag_kind)
    upper_plot = get_plot_fn(upper_kind)
    lower_plot = get_plot_fn(lower_kind)

    pg.map_diag(diag_plot)
    pg.map_upper(upper_plot)
    pg.map_lower(lower_plot)

    # add legend
    pg.add_legend()

    # save plot
    pg.savefig(dest_path)

