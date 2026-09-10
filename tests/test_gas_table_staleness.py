from datetime import date, timedelta

from scoring import (
    GAS_TABLE_LAST_REVIEWED,
    GAS_TABLE_REVIEW_INTERVAL_DAYS,
    _warn_if_gas_table_stale,
)


def test_no_warning_when_recently_reviewed(capsys):
    today = GAS_TABLE_LAST_REVIEWED + timedelta(days=GAS_TABLE_REVIEW_INTERVAL_DAYS)
    _warn_if_gas_table_stale(today=today)
    assert capsys.readouterr().err == ""


def test_warns_once_interval_exceeded(capsys):
    today = GAS_TABLE_LAST_REVIEWED + timedelta(days=GAS_TABLE_REVIEW_INTERVAL_DAYS + 1)
    _warn_if_gas_table_stale(today=today)
    err = capsys.readouterr().err
    assert "警告" in err
    assert str(GAS_TABLE_LAST_REVIEWED) in err
