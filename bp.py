import numpy as np

from data import x_train, x_test, y_train, y_test


class NeuralNetwork():
    """
    this is a basic class of neural network
    """

    def __init__(self, layers):
        # first we assume this network contains
        # only 3 layers: input, hidden, out
        self.n_input = layers[0]
        self.n_hidden = layers[1]
        self.n_out = layers[2]
        self.weight_ih = np.random.rand(self.n_input, self.n_hidden)
        self.weight_ho = np.random.rand(self.n_hidden, self.n_out)
        self.delta = [[] for col in range(2)]

    def train(self, x_train, y_train):
        """
        :param x_train: 2d array, features
        :param y_train: 1d array, labels of x_train
        :return: None
        """
        count = 0
        for x_entries in x_train:
            y_entries = [0 for col in range(self.n_out)]
            y_entries[y_train[count]] = 1
            count += 1
            # x_entries = 1d array, y_entries = 1d array
            h_in = np.dot(np.transpose(self.weight_ih), x_entries)
            h_out = list(map(self.sigmoid, h_in))
            o_in = np.dot(np.transpose(self.weight_ho), h_out)
            o_out = list(map(self.sigmoid, o_in))
            self.delta[1] = np.array(o_out) - np.array(y_entries)
            self.delta[0] = np.dot(self.weight_ho, self.delta[1])
            mul_h = np.multiply(np.multiply(o_out, np.array([1 for col in range(self.n_out)]) - np.array(o_out)), h_out)
            mul_h = np.multiply(np.multiply(mul_h, 0.3), self.delta[1])
            self.weight_ho = self.weight_ho - mul_h
            mul_i = np.multiply(np.multiply(h_out, np.array([1 for col in range(self.n_hidden)])
                                            - np.array(h_out)), x_entries)
            mul_i = np.multiply(np.multiply(mul_i, 0.3), self.delta[0])
            self.weight_ho = self.weight_ho - mul_h
            self.weight_ih = self.weight_ih - mul_i

    def sigmoid(self, value):
        """

        :param value: scalar value in a 1d array
        :return: sigmoid value of the input
        """
        return 1 / (1 + np.power(np.e, -value))

    def predict(self, x_test):
        """
        predict the result of x_test
        :param x_test: 2d array, features
        :return: 1d array, the prediction of x_test
        """
        y_test = []
        for sample in x_test:
            # todo
            # gen prediction of each sample
            # y_test.appen()
            h_in = np.dot(np.transpose(self.weight_ih), sample)
            h_out = list(map(self.sigmoid, h_in))
            o_in = np.dot(np.transpose(self.weight_ho), h_out)
            o_out = list(map(self.sigmoid, o_in))
            y = list(map(self.thresh, o_out))
            y = np.dot(y, [i for i in range(10)])
            y_test.append(y)
        return y_test

    def thresh(self, x):
        """

        :param x: input value of a 1d array
        :return: 1d array with 0 or 1 if < threshold or > threshold respectively
        """
        if x > 0.5:
            return 1
        else:
            return 0

if __name__ == '__main__':
    nn = NeuralNetwork([784, 30, 10])
    nn.train(x_train, y_train)
    predicty = nn.predict(x_test)
    print(np.sum(predicty == y_test))
