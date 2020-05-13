from core.md_parser import extract_nlu, nlu2md, _extract_h2s
from core.util import open_file


def test_extract_nlu():
    result = extract_nlu(open_file('examples/nlu_no_entities.md'))
    assert result is not None


def test_nlu2md():
    result = nlu2md([('## intent: chitchat/what_should_i_buy', ['what should I buy','tell me what I should buy?'])])
    assert result is not None
    assert result == "\n\n## intent: chitchat/what_should_i_buy\n- what should I buy\n- tell me what I should buy?"


def test_extract_hs2():
    md = open_file('examples/nlu_retreival_intents.md')
    result = _extract_h2s(md)
    assert result == ['## intent: chitchat/what_should_i_buy', '## intent: how_should_i_spend_money', '## intent: chitchat/tell_me_a_joke']
