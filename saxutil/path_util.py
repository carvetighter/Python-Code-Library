import os


def basepath():
    end = None
    pfx = os.path.realpath(__file__)
    while end not in {'lib'}:
        pfx, end = os.path.split(pfx)
    return os.path.join(pfx, end)


def data_dir():
    return os.path.join(basepath(), 'data')
