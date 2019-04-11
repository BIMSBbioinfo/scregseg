
from scseg import longest
from scseg.cli import main


def test_main():
    main([])


def test_longest():
    assert longest([b'a', b'bc', b'abc']) == b'abc'
