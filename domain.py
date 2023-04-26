from collections import Iterable


class Domain(object):
    def __init__(self, lower_bound, upper_bound):
        if not isinstance(lower_bound, Iterable):
            lower_bound = (lower_bound,)

        if not isinstance(upper_bound, Iterable):
            upper_bound = (upper_bound,)

        assert len(lower_bound) == len(upper_bound)

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.dim = len(self.lower_bound)
