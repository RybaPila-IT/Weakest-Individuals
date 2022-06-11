from hints.aliases import ObjectiveFunction


def negate(f: ObjectiveFunction) -> ObjectiveFunction:
    """
    Negate the original function f

    The negation means changing the function output by multiplying it
    with -1 value.

    :param f: function which will be negated

    :return: Negated function f
    """
    return lambda x: -f(x)
