class perceptron:

    def __init__(self, size: int, learn_rate: float) -> None:
        self.weights = [1. for i in range(size)]
        self.learn_rate = learn_rate

    def classify(self, values: list[float]) -> int:
        if len(values) != len(self.weights):
            raise Exception("incorrect length of values")
        sum = 0
        for i in range(len(self.weights)):
            sum += self.weights[i] * values[i]
        if sum > 0:
            return 1
        return -1

    def train(self, values: list[float], expect: int) -> None:
        if self.classify(values) != expect:
            for i in range(len(self.weights)):
                self.weights[i] = self.weights[i] + self.learn_rate * expect * values[i]

    def bulk_train(self, value_list: list[list[float]], expects: list[int]) -> None:
        if len(value_list) != len(expects):
            raise Exception("lengths of the value_list and expects do not correspond")
        for i in range(len(value_list)):
            self.train(value_list[i], expects[i])

    def evaluate(self, value_list: list[list[float]], expects: list[int]) -> int:
        if len(value_list) != len(expects):
            raise Exception("lengths of the value_list and expects do not correspond")
        wins = 0
        for i in range(len(value_list)):
            if self.classify(value_list[i]) == expects[i]:
                wins += 1
        return wins
