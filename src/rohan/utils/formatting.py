"""Canonical monetary formatting utilities.

All monetary values in ROHAN follow the ABIDES convention: integers in
**cents** (e.g. ``18550`` = $185.50).  These helpers convert cents to
human-readable dollar strings in a consistent way across the codebase,
replacing the previously scattered inline helpers:

* ``_fmt_dollar()``   in ``src/rohan/framework/prompts.py``
* ``_m_dollar()`` / ``_mv_dollar()``  in ``src/rohan/ui/0_Terminal.py``
"""

from __future__ import annotations


def fmt_dollar(cents: int | float, precision: int = 2) -> str:
    """Format a value in cents as a signed dollar string.

    Examples
    --------
    >>> fmt_dollar(18550)
    '$185.50'
    >>> fmt_dollar(-300)
    '-$3.00'
    >>> fmt_dollar(0)
    '$0.00'
    >>> fmt_dollar(1234567, precision=0)
    '$12346'
    """
    dollars = cents / 100.0
    sign = "-" if dollars < 0 else ""
    return f"{sign}${abs(dollars):,.{precision}f}"


def fmt_dollar_metric(cents: int | float | None, precision: int = 2) -> str:
    """Format a value in cents for ``st.metric()`` display (no currency symbol, signed).

    Returns ``"N/A"`` when *cents* is ``None``.

    Examples
    --------
    >>> fmt_dollar_metric(18550)
    '185.50'
    >>> fmt_dollar_metric(-300)
    '-3.00'
    >>> fmt_dollar_metric(None)
    'N/A'
    """
    if cents is None:
        return "N/A"
    return f"{cents / 100.0:,.{precision}f}"


def fmt_dollar_delta(cents: int | float | None, precision: int = 2) -> str:
    """Format a delta value for ``st.metric(delta=...)`` display.

    Always includes an explicit sign so Streamlit can colour it correctly.
    Returns ``"0.00"`` when *cents* is ``None``.

    Examples
    --------
    >>> fmt_dollar_delta(500)
    '+5.00'
    >>> fmt_dollar_delta(-200)
    '-2.00'
    >>> fmt_dollar_delta(None)
    '0.00'
    """
    if cents is None:
        return "0.00"
    dollars = cents / 100.0
    sign = "+" if dollars >= 0 else ""
    return f"{sign}{dollars:,.{precision}f}"
