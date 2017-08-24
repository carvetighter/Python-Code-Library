import io
import csv


def _is_var_csv_serializable(x):
    if not isinstance(x, str) and hasattr(x, '__iter__'):
        return False
    return True


def _serialize_err_check(k, v):
    if not _is_var_csv_serializable(k):
        raise ValueError('KEY is not serializable in csv')
    if not _is_var_csv_serializable(v):
        raise ValueError('VALUE is not serializable in csv')


def csv_dumps(d, delimiter=','):
    """
    :type d: dict
    :type delimiter: str
    """
    si = io.StringIO()
    wtr = csv.writer(si, delimiter=delimiter)
    for k, v in d.items():
        _serialize_err_check(k, v)
        wtr.writerow([k, v])
    return si.getvalue().strip()


def csv_loads(dta, delimiter=',', ktype=str, vtype=str):
    """
    :type dta: str
    :type delimiter: str
    :type ktype: type
    :type vtype: type
    """
    d = {}
    si = io.StringIO(dta)
    rdr = csv.reader(si, delimiter=delimiter)
    for row in rdr:
        d[ktype(row[0])] = vtype(row[1])
    return d
