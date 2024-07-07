"""Utilities for qablet models"""

import numpy as np
import pandas as pd
import pyarrow as pa
from qablet.base.flags import Stats
from qablet.base.utils import discounter_from_dataset
from qablet_contracts.timetable import TS_EVENT_SCHEMA, py_to_ts


def option_prices(ticker, expirations, strikes, is_call, model, ref_dataset):
    """Get option prices for a range of expirations and strikes
    in a single simulation."""

    # Create a timetable that pays forwards at each expiration
    events = [
        {
            "track": "",
            "time": dt,
            "op": "+",
            "quantity": 1,
            "unit": ticker,
        }
        for dt in expirations
    ]

    events_table = pa.RecordBatch.from_pylist(events, schema=TS_EVENT_SCHEMA)
    fwd_timetable = {"events": events_table, "expressions": {}}

    dataset = ref_dataset.copy()
    discounter = discounter_from_dataset(dataset)
    dataset["MC"]["FLAGS"] = Stats.CASHFLOW

    _, stats = model.price(fwd_timetable, dataset)
    # cashflows for track 0, all events
    cf = stats["CASHFLOW"][0]

    price_df = pd.DataFrame.from_dict({"Strike": strikes})

    for i, exp in enumerate(expirations):
        prc_ts = dataset["PRICING_TS"]
        # Get Time in years from the millisecond timestamps
        T = (py_to_ts(exp).value - prc_ts) / (365.25 * 24 * 3600 * 1e3)
        df = discounter.discount(T)

        # calculate prices (value as of expiration date)
        event_cf = cf[i] / df
        strikes_c = strikes[..., None]  # Turn into a column vector
        pay = event_cf - strikes_c
        prices = np.maximum(pay, 0).mean(axis=1)

        exp_str = exp.strftime("%Y-%m-%d")
        price_df[exp_str] = prices * df

    return price_df
