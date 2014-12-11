__author__ = 'thunderlbc'
import os
import time
import sys

import numpy

import theano
import theano.tensor as T

from logistic_sgd import LogisticRegression, load_data


class HiddenLayer(object):

    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        self.input = input
        if W is None:
            W_value = numpy.asarray(
                rng.uniform(
                    low=numpy.sqrt(6./(n_in + n_out)),
                    high=numpy.sqrt(6./(n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.softmax:
                W_value *= 4.
            W = theano.shared(value=W_value, name='W', borrow=True)
        if b is None:
            b_value = numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            )
            b = theano.shared(value=b_value, name='b', borrow=True)
        self.W = W
        self.b = b
        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.params = [self.W, self.b]


class MLP(object):

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        self.hiddenLayer = HiddenLayer(rng=rng, input=input, n_in=n_in, n_out=n_hidden, activation=T.tanh)
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )
        self.L1 = (abs(self.hiddenLayer.W).sum() + abs(self.logRegressionLayer.W).sum())
        self.L2_sqr = ((self.hiddenLayer.W ** 2).sum() + (self.logRegressionLayer.W ** 2).sum())
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        self.errors = self.logRegressionLayer.errors
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params


def test_MLP(learning_rate=0.01, L1_reg=0.00, L2_reg=0.001, n_epochs=1000, dataset='mnist.pkl.gz', batch_size=20, n_hidden=500):

    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    print '...Building the model.'

    index = T.iscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    rng = numpy.random.RandomState(1234)

    classifier = MLP(
        rng=rng,
        input=x,
        n_in=28*28,
        n_hidden=n_hidden,
        n_out=10
    )
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1 +
        L2_reg * classifier.L2_sqr
    )

    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )
    valid_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )
    gparams = [T.grad(cost, param) for param in classifier.params]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size:(index + 1) * batch_size],
            y: train_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    print '..training.'

    patience = 10000
    patience_inc = 2
    improvement_threshold = 0.995
    validation_freq = min(patience / 2, n_train_batches)

    best_validation_loss = numpy.inf
    best_iter = 0
    best_score = 0
    start_time = time.time()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in xrange(n_train_batches):
            avg_train_cost = train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if (iter + 1) / validation_freq == 0:
                validation_loss = [
                    valid_model(i)
                    for i in xrange(n_valid_batches)
                ]
                this_valid_loss = numpy.mean(validation_loss)
                print (
                    'ecoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_valid_loss * 100
                    )
                )
                if this_valid_loss < best_validation_loss:
                    if(
                        this_valid_loss < improvement_threshold * best_validation_loss
                    ):
                        patience = max(patience, iter * patience_inc)
                    best_validation_loss = this_valid_loss
                    best_iter = iter

                    test_loss = [test_model(i) for i in xrange(n_test_batches)]
                    best_score = numpy.mean(test_loss)
                    print(
                        'epoch %i, minibatch %i/%i, test error %f %%' %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            best_score * 100
                        )
                    )
            if patience <= iter:
                done_looping = True
                break

    end_time = time.time()
    print(
        'Optimization Complete! Best Validation Score %f %%'
        'obtained at iteration %i, with test performance %f %%' %
        (
            best_validation_loss * 100,
            best_iter + 1,
            best_score * 100
        )
    )
    print >>sys.stderr, ('The code for file' + os.path.split(__file__)[1]+'ran for%.2fm' % ((end_time-start_time)/60.))

if __name__ == "__main__":
    test_MLP()





