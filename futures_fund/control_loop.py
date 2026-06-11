from __future__ import annotations

from datetime import datetime

from futures_fund.models import Cadence
from futures_fund.scheduling import cycle_due

_CADENCE_TF = {"weekly": 7 * 1440, "daily": 1440}  # 10080 / 1440 minutes


def cadence_due(state_dir, now_utc: datetime, cadence: Cadence) -> tuple[str, int, str]:
    tf = _CADENCE_TF[cadence]
    # loop=cadence => cycle root state/<cadence>/cycle/* (matches cycle_io.cycle_dir)
    return cycle_due(state_dir, now_utc, tf_minutes=tf, loop=cadence)
