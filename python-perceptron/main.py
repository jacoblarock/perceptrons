from perceptron import perceptron
from random import randint

if __name__ == "__main__":
    data = []
    expects = []
    for i in range(25000000):
        x = randint(-50, 50)
        y = randint(-50, 50)
        data.append([1, x, y])
        if y > 0:
            expects.append(1)
        else:
            expects.append(-1)
    p = perceptron(3, 0.2)
    print(p.evaluate(data, expects))
    print(p.weights)
    p.bulk_train(data, expects)
    print(p.evaluate(data, expects))
    print(p.weights)
    p.bulk_train(data, expects)
    print(p.evaluate(data, expects))
    print(p.weights)
    p.bulk_train(data, expects)
    print(p.evaluate(data, expects))
    print(p.weights)
