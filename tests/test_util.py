from core.util import open_file


def test_open_file():
    result = open_file("examples/nlu_no_entities.md")
    assert result is not None
