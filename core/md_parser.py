import re
from typing import List

from core.util import open_file


def _extract_h2s(md: str) -> list:
    return re.findall(r"(##.*)", md)


def extract_nlu(md: str) -> List[tuple]:
    h2s: list = _extract_h2s(md)
    utterances: list = []
    intent: list = [x.strip() for x in md.split("##") if x]
    for item in intent:
        utterances.append([items for _, items in re.findall(r"(!?\-\s)(.*)", item)])

    return list(zip(h2s, utterances))


def nlu2md(nlu: List[tuple]):
    nlu_md: str = ""
    for h2, utterances in nlu:
        nlu_md += f"\n\n{h2}"
        for u in utterances:
            # remove any double quotes
            u = re.sub(r"(``|\'\'|\"\")", "", u)
            nlu_md += f"\n- {u}"
    return nlu_md
