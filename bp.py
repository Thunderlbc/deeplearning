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
        self.weight_ih = np.random.randn(self.n_input + 1, self.n_hidden)  # weight matrix for layer Input and layer Hid
        self.weight_ho = np.random.randn(self.n_hidden + 1, self.n_out)  # weight matrix for layer Hid and output
        self.delta = [[] for col in range(2)]

    def train(self, x_train, y_train):
        """
        :param x_train: 2d array, features
        :param y_train: 1d array, labels of x_train
        :return: None
        """
        order = [i for i in range(60000)]
        for i in range(15):
            np.random.shuffle(order)
            count = 0
            print('Iteration Times:')
            print(i)
            weight_gradient_ih_accu = np.zeros((self.n_input + 1, self.n_hidden))
            weight_gradient_ho_accu = np.zeros((self.n_hidden + 1, self.n_out))
            for j in range(60000):
                x_entries = np.hstack((x_train[order[j]], 1))
                y_entries = [0 for col in range(self.n_out)]
                y_entries[y_train[order[j]]] = 1
                #accumulate the gradient for 100 times
                count += 1
                # x_entries = 1d array, y_entries = 1d array
                h_in = np.dot(np.transpose(self.weight_ih), x_entries)
                h_out_less = list(map(self.sigmoid, h_in))
                h_out = np.hstack((h_out_less, np.array(1)))
                o_in = np.dot(np.transpose(self.weight_ho), h_out)
                o_out = list(map(self.sigmoid, o_in))
                #Update the remaining difference for each layer
                self.delta[1] = np.array(o_out) - np.array(y_entries)
                self.delta[0] = np.dot(self.weight_ho, self.delta[1])
                #Compute the gradient of each weight matrix
                mul_h = np.multiply(o_out, np.array([1 for col in range(self.n_out)]) - np.array(o_out))
                mul_h = np.dot(np.array(h_out).reshape(self.n_hidden + 1, 1), np.array(mul_h).reshape(1, self.n_out))
                mul_h = np.multiply(np.multiply(mul_h, 0.3), self.delta[1])
                #
                mul_i = np.multiply(h_out_less, np.array([1 for col in range(self.n_hidden)]) - np.array(h_out_less))
                mul_i = np.multiply(np.multiply(mul_i, 0.3), self.delta[0][0:self.n_hidden])
                mul_i = np.dot(np.array(x_entries).reshape(self.n_input + 1, 1),
                               np.array(mul_i).reshape(1, self.n_hidden))
                #accumulate the gradient for a proper times then update the weight matrix
                weight_gradient_ho_accu = weight_gradient_ho_accu + mul_h
                weight_gradient_ih_accu = weight_gradient_ih_accu + mul_i
                #mini-Batch
                if count == 19:
                    count = 0
                    weight_gradient_ho_accu = np.divide(weight_gradient_ho_accu, 20)
                    weight_gradient_ih_accu = np.divide(weight_gradient_ih_accu, 20)
                    self.weight_ho = self.weight_ho - weight_gradient_ho_accu
                    self.weight_ih = self.weight_ih - weight_gradient_ih_accu
                    weight_gradient_ih_accu = np.zeros((self.n_input + 1, self.n_hidden))
                    weight_gradient_ho_accu = np.zeros((self.n_hidden + 1, self.n_out))

    def sigmoid(self, value):
        """
        :param value: scalar value in a 1d array
        :return: sigmoid value of the input
        """
        return 1. / (1. + np.exp(-value))

    def predict(self, x_test):
        """
        predict the result of x_test
        :param x_test: 2d array, features
        :return: 1d array, the prediction of x_test
        """
        y_predict = []
        for sample in x_test:
            # todo
            # gen prediction of each sample
            # y_test.appen()
            #print(sample)
            h_in = np.dot(np.transpose(self.weight_ih), np.hstack((sample, np.array(1))))
            h_out = np.hstack((list(map(self.sigmoid, h_in)), np.array(1)))
            o_in = np.dot(np.transpose(self.weight_ho), h_out)
            o_out = list(map(self.sigmoid, o_in))
            #print(o_out)
            #y = list(map(self.thresh, o_out))
            #y = np.dot(y, [i for i in range(10)])
            y_re = np.argmax(o_out)
            y_predict.append(y_re)
        return y_predict


if __name__ == '__main__':
    nn = NeuralNetwork([784, 30, 10])
    nn.train(x_train, y_train)
    predicty = nn.predict(x_test)
    print(predicty)
    print(np.sum(predicty == y_test))
