"""Test helper: named fields of a result, as a tuple, in the order asked."""


def fields_of(result, *names):
    return tuple(getattr(result, n) for n in names)
