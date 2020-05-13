import re

def open_file(path: str) -> str:
    with open(path) as f:
        return f.read()


def extract(md: str, pattern: str) -> list:
    return [ items for _, items in re.findall(pattern, md)]


def extract_nlu(path: str):
    # TODO group by h2
    return extract(open_file(path), pattern = r'(!?\-\s)(.*)')


if __name__ == '__main__':
    import argparse
    from time import perf_counter

    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--md', help="A markdown file")
    args = parser.parse_args()

    print(extract_nlu('examples/nlu_no_entities.md'))
