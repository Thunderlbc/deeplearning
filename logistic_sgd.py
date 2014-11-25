__author__ = 'thunderlbc'
import cPickle
import os
import gzip
import sys
import time

import numpy
import theano
import theano.tensor as T


class LogisticRegression(object):

    def __init__(self, input, n_in, n_out):
        """
        Initiation of LogisticRegression class. W and b are both theano.shared value.
        :param input: a chunk of x or a batch of x.
        :param n_in:  the dimension of the element in input.
        :param n_out: the dimension of the output vector
        :return:
        """
        self.W = theano.shared(value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX), name='W', borrow=True)
        self.b = theano.shared(value=numpy.zeros((n_out,), dtype=theano.config.floatX), name='b', borrow=True)
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """
        Return negative log likelihood value of the given label y
        :param y:
        :return: negative log likelihood value
        """
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """
        Compute the error for validation. T.neq is for not equal.
        :param y:
        :return:
        """
        # y and y_pred must have the same dimension
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred', ('y', y.type, 'y_pred', self.y_pred.type))
        #the dtype of y must be one of the int types.
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


def load_data(dataset):
    """
    Load data from the current file path or the Internet(url not work now~)
    :param dataset: where the data loaded
    :return: three tuple of shared datasets.
    """
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        new_path = os.path.join(os.path.split(__file__)[0], "..", "data", dataset)
        if os.path.isfile(new_path) and data_file == "mnist.pkl.gz":
            dataset = new_path
    if not os.path.isfile(dataset) and data_file == "mnist.pkl.gz":
        import urllib
        origin = ('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz')
        #http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)
    print 'Loading data...'
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    def shared_dataset(data_xy, borrow=True):
        """
        Create two corresponding shared data using data_xy.
        :param data_xy: the input used for creating the two shared data.
        :param borrow: default to be true. i.e. does not copy the value of the data_xy.
        :return: two shared dataset derived from data_xy
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    test_x, test_y = shared_dataset(test_set)
    train_x, train_y = shared_dataset(train_set)
    valid_x, valid_y = shared_dataset(train_set)

    rval = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
    return rval


def sgd_optimization_mnist(learning_rate=0.3, n_epochs=1000, dataset='mnist.pkl.gz', batch_size=600):
    """
    Optimization Process using the SGD algorithm.
    :param learning_rate: default to be 0.3.
    :param n_epochs:  the number of times that the dataset to be scaned
    :param dataset:  where the data was loaded.
    :param batch_size: the pre-defined number of a chunk size
    :return:
    """
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    #compute the number of batches of train_set, valid_set, test_set.
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    print 'Building the model...'

    index = T.iscalar()

    x = T.matrix('x')
    y = T.ivector('y')

    clasifier = LogisticRegression(input=x, n_in=28*28, n_out=10)

    cost = clasifier.negative_log_likelihood(y)

    #with givens parameter, each time this function called, the x and y will be replaced with the expression.
    test_model = theano.function(
        inputs=[index],
        outputs=clasifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    #with givens parameter, each time this function called, the x and y will be replaced with the expression.
    valid_model = theano.function(
        inputs=[index],
        outputs=clasifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    g_W = T.grad(cost=cost, wrt=clasifier.W)
    g_b = T.grad(cost=cost, wrt=clasifier.b)

    updates=[(clasifier.W, clasifier.W - learning_rate * g_W), (clasifier.b, clasifier.b - learning_rate * g_b)]

    #with givens parameter, each time this function called, the x and y will be replaced with the expression.
    #with updates parameter, each time this function called, W and b will be updated by the expression above.
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size:(index + 1) * batch_size],
            y: train_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    print 'Training Model...'

    #At least check this many examples.
    patience = 5000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = numpy.inf
    test_score = 0
    start_time = time.clock()

    done_looping = False
    epoch = 0

    while epoch < n_epochs and not done_looping:
        epoch += 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if (iter + 1) % validation_frequency == 0:
                validation_losses = [valid_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print (
                    'Epoch %i, minibatch %i/%i, validation error %f%%'
                    % (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100
                    )
                )

                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                    best_validation_loss = this_validation_loss

                    test_losses = [test_model(l) for l in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    print (
                        'Epoch %i, minibatch %i/%i, test error of best model %f%%'
                        % (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100
                        )
                    )
                if patience <= iter:
                    done_looping = True
                    break
    end_time = time.clock()
    print(
        'Optimization complete with best validation score of %f%%,'
        'with test performence %f%%'
        % (
            best_validation_loss * 100,
            test_score * 100
        )
    )
    print(
        'The code run for %d epochs, with %f epochs/secs'
        % (
            epoch,
            1. * epoch/(end_time - start_time)
        )
    )
    print >> sys.stderr, ('The code for file ' +
                           os.path.split(__file__)[1] +
                           ' ran for %.1fs' % (end_time - start_time))

if __name__ == '__main__':
    sgd_optimization_mnist()
