import re

def filename(path: str) -> str:
    return re.sub(r'(\.)(\w.*)', '', path)


def file_ext(path: str) -> str:
    return "".join(re.findall(r'(\.\w.*)', path))


def open_file(path: str) -> str:
    with open(path) as f:
        return f.read()
