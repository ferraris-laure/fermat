
class DistanceCalculatorMethod:

    def __init__(self, alpha, **kwargs) -> None:
        super().__init__()
        self.alpha = alpha

    def fit(self, distances):
        raise NotImplementedError()  # pragma: no cover

    def get_distance(self, a, b):
        raise NotImplementedError()  # pragma: no cover

    def get_distances(self):
        raise NotImplementedError()  # pragma: no cover
