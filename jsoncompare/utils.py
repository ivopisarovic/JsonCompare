def is_dict(o):
    return type(o) is dict


def is_not_dict(o):
    return not is_dict(o)


def is_bool(o):
    return type(o) is bool


def is_not_bool(o):
    return not is_bool(o)


def is_int(o):
    return type(o) is int


def is_not_int(o):
    return not is_int(o)


def is_float(o):
    return type(o) is float


def is_not_float(o):
    return not is_float(o)


def is_str(o):
    return type(o) is str


def is_not_str(o):
    return not is_str(o)


def is_list(o):
    return type(o) is list


def is_not_list(o):
    return not is_list(o)


def types_is_equal(a, b):
    return type(a) is type(b)


def types_is_not_equal(a, b):
    return not types_is_equal(a, b)


def is_primitive(o):
    return type(o) in (int, float, bool, str)


def is_not_primitive(o):
    return not is_primitive(o)


def is_iterable(o):
    return type(o) in (dict, list, tuple, set)


def is_not_iterable(o):
    return not is_iterable(o)
