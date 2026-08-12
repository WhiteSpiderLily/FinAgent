"""Tests for akshare_src Session.__init__ UA patch."""
import requests


def test_session_gets_browser_ua():
    """_session_init_with_ua replaces default python-requests UA with browser UA."""
    from finagent.sources.akshare_src import _session_init_with_ua, _AKSHARE_UA

    s = requests.Session.__new__(requests.Session)
    _session_init_with_ua(s)
    ua = s.headers.get("User-Agent", "")
    assert "python-requests/" not in ua
    assert ua == _AKSHARE_UA


def test_session_init_is_patched():
    """Importing akshare_src patches requests.Session.__init__."""
    import finagent.sources.akshare_src  # noqa: F401
    assert requests.Session.__init__.__name__ == "_session_init_with_ua"
