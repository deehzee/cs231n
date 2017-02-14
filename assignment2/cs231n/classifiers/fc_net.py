from __future__ import absolute_import, division, print_function

import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity
    and softmax loss that uses a modular layer design. We assume an
    input dimension of D, a hidden dimension of H, and perform
    classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead,
    it will interact with a separate Solver object that is
    responsible for running optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100,
                 num_classes=10, weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to
          classify
        - weight_scale: Scalar giving the standard deviation for
          random initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ##############################################################
        # TODO: Initialize the weights and biases of the two-layer   #
        # net. Weights should be initialized from a Gaussian with    #
        # standard deviation equal to weight_scale, and biases       #
        # should be initialized to zero. All weights and biases      #
        # should be stored in the dictionary self.params, with first #
        # layer weights and biases using the keys 'W1' and 'b1' and  #
        # second layer weights and biases using the keys 'W2' and    #
        # 'b2'.                                                      #
        ##############################################################
        self.D = input_dim
        self.H = hidden_dim
        self.C = num_classes
        W1 = weight_scale * np.random.randn(self.D, self.H)
        W2 = weight_scale * np.random.randn(self.H, self.C)
        b1 = np.zeros(self.H)
        b2 = np.zeros(self.C)
        self.params.update({'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2})

        ##############################################################
        #                      END OF YOUR CODE                      #
        ##############################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for
          X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model
        and return:
        - scores: Array of shape (N, C) giving classification scores,
          where scores[i, c] is the classification score for X[i] and
          class c.

        If y is not None, then run a training-time forward and
        backward pass and return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping
          parameter names to gradients of the loss with respect to
          those parameters.
        """
        scores = None
        ##############################################################
        # TODO: Implement the forward pass for the two-layer net,    #
        # computing the class scores for X and storing them in the   #
        # scores variable.                                           #
        ##############################################################
        
        # X1 := output of affine+relu on X
        # X2 := ouyput of affine on X1

        # Get the params
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']

        X1, cache1 = affine_relu_forward(X, W1, b1)
        X2, cache2 = affine_forward(X1, W2, b2)

        scores = X2
        ##############################################################
        #                      END OF YOUR CODE                      #
        ##############################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ##############################################################
        # TODO: Implement the backward pass for the two-layer net.   #
        # Store the loss in the loss variable and gradients in the   #
        # grads dictionary. Compute data loss using softmax, and     #
        # make sure that grads[k] holds the gradients for            #
        # self.params[k]. Don't forget to add L2 regularization!     #
        #                                                            #
        # NOTE: To ensure that your implementation matches ours and  #
        # you pass the  automated tests, make sure that your L2      #
        # regularization includes a factor of 0.5 to simplify the    #
        # expression for the gradient.                               #
        ##############################################################
        loss, dX2 = softmax_loss(X2, y)
        dX1, dW2, db2 = affine_backward(dX2, cache2)
        dX, dW1, db1 = affine_relu_backward(dX1, cache1)

        # Regularize
        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        dW1 += self.reg * W1
        dW2 += self.reg * W2

        grads.update({'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2})
        ##############################################################
        #                      END OF YOUR CODE                      #
        ##############################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of 
    idden layers, ReLU nonlinearities, and a softmax loss function.
    This will also implement dropout and batch normalization as
    options. For a network with L layers, the architecture will be

    {affine - [batch norm] - relu - [dropout]} \
        x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...}
    block is repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored
    in the self.params dictionary and will be learned using the
    Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32,
                 num_classes=10, loss_func='softmax',
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each
          hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to
          classify.
        - loss_func: [djn] Loss function ('softmax' or 'svm').
        - dropout: Scalar between 0 and 1 giving dropout strength. If
          dropout=0 then the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch
          normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for
          random initialization of the weights.
        - dtype: A numpy datatype object; all computations will be
          performed using this datatype. float32 is faster but less
          accurate, so you should use float64 for numeric gradient
          checking.
        - seed: If not None, then pass this random seed to the dropout
          layers. This will make the dropout layers deteriminstic so
          we can gradient check the model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        
        ##### [djn] #####
        loss_funcs = {
            'softmax': softmax_loss,
            'svm': svm_loss,
        }
        if loss_func not in loss_funcs:
            raise ValueError('Unknown loss function: {}'.format(
                loss_func))
        self.loss_func = loss_funcs[loss_func]


        ##############################################################
        # TODO: Initialize the parameters of the network, storing    #
        # all values in the self.params dictionary. Store weights    #
        # and biases for the first layer in W1 and b1; for the       #
        # second layer use W2 and b2, etc. Weights should be         #
        # initialized from a normal distribution with standard       #
        # deviation equal to weight_scale and biases should be       #
        # initialized to zero.                                       #
        #                                                            #
        # When using batch normalization, store scale and shift      #
        # parameters for the first layer in gamma1 and beta1; for    #
        # the second layer use gamma2 and beta2, etc. Scale          #
        # parameters should be initialized to one and shift          #
        # parameters should be initialized to zero.                  #
        ##############################################################
        dims = [input_dim]
        dims.extend(hidden_dims)
        dims.append(num_classes)

        for i in xrange(0, self.num_layers):
            W = weight_scale * np.random.randn(dims[i], dims[i + 1])
            b = np.zeros(dims[i + 1])
            self.params['W{}'.format(i + 1)] = W
            self.params['b{}'.format(i + 1)] = b
            if self.use_batchnorm and i < self.num_layers - 1:
                gamma = np.ones(dims[i + 1])
                beta = np.zeros(dims[i + 1])
                self.params['gamma{}'.format(i + 1)] = gamma
                self.params['beta{}'.format(i + 1)] = beta

        self.dims = dims
        ##############################################################
        #                      END OF YOUR CODE                      #
        ##############################################################

        # When using dropout we need to pass a dropout_param
        # dictionary to each dropout layer so that the layer knows the
        # dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each
        # dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running
        # means and variances, so we need to pass a special bn_param
        # object to each batch normalization layer. You should pass
        # self.bn_params[0] to the forward pass of the first batch
        # normalization layer, self.bn_params[1] to the forward pass
        # of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'}
                              for i in xrange(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param
        # since they behave differently during training and testing.
        ##if self.dropout_param is not None:
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ##############################################################
        # TODO: Implement the forward pass for the fully-connected   #
        # net, computing the class scores for X and storing them in  #
        # the scores variable.                                       #
        #                                                            #
        # When using dropout, you'll need to pass self.dropout_param #
        # to each dropout forward pass.                              #
        #                                                            #
        # When using batch normalization, you'll need to pass        #
        # self.bn_params[0] to the forward pass for the first batch  #
        # normalization layer, pass self.bn_params[1] to the forward #
        # pass for the second batch normalization layer, etc.        #
        ##############################################################
        caches = []
        inp = X     # input
        out = None  # output

        # Iterate over each hidden layers
        for i in xrange(self.num_layers - 1):
            # affine layer
            W = self.params['W{}'.format(i + 1)]
            b = self.params['b{}'.format(i + 1)]
            out, cache = affine_forward(inp, W, b)
            caches.append(cache)
            inp = out  # for the next step

            # batchnorm layer
            if self.use_batchnorm:
                bn_param = self.bn_params[i]
                gamma = self.params['gamma{}'.format(i + 1)]
                beta = self.params['beta{}'.format(i + 1)]
                out, cache = batchnorm_forward(inp, gamma, beta,
                                               bn_param)
                caches.append(cache)
                inp = out

            # relu layer
            out, cache = relu_forward(inp)
            caches.append(cache)
            inp = out 

            # dropout layer
            if self.use_dropout:
                out, cache = dropout_forward(inp, self.dropout_param)
                caches.append(cache)
                inp = out

        # final affine layer
        W = self.params['W{}'.format(self.num_layers)]
        b = self.params['b{}'.format(self.num_layers)]
        out, cache = affine_forward(inp, W, b)
        caches.append(cache)
        scores = out

        ##############################################################
        #                      END OF YOUR CODE                      #
        ##############################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ##############################################################
        # TODO: Implement the backward pass for the fully-connected  #
        # net. Store the loss in the loss variable and gradients in  #
        # the grads dictionary. Compute data loss using softmax, and #
        # make sure that grads[k] holds the gradients for            #
        # self.params[k]. Don't forget to add L2 regularization!     #
        #                                                            #
        # When using batch normalization, you don't need to          #
        # regularize the scale and shift parameters.                 #
        #                                                            #
        # NOTE: To ensure that your implementation matches ours and  #
        # you pass the automated tests, make sure that your L2       #
        # regularization includes a factor of 0.5 to simplify the    #
        # expression for the gradient.                               #
        ##############################################################
        loss, dx = self.loss_func(scores, y)

        # Last affine layer
        cache = caches.pop()
        dx, dw, db = affine_backward(dx, cache)
        grads['W{}'.format(self.num_layers)] = dw
        grads['b{}'.format(self.num_layers)] = db

        # Iterate over hidden layers backwards
        for i in xrange(self.num_layers - 1, 0, -1):
            # dropout layer
            if self.use_dropout:
                cache = caches.pop()
                dx = dropout_backward(dx, cache)

            # relu layer
            cache = caches.pop()
            dx = relu_backward(dx, cache)

            # batchnorm layer
            if self.use_batchnorm:
                cache = caches.pop()
                dx, dgamma, dbeta = batchnorm_backward_alt(dx, cache)
                grads['gamma{}'.format(i)] = dgamma
                grads['beta{}'.format(i)] = dbeta

            # affine layer
            cache = caches.pop()
            dx, dw, db = affine_backward(dx, cache)
            grads['W{}'.format(i)] = dw
            grads['b{}'.format(i)] = db

        ##############################################################
        #                      END OF YOUR CODE                      #
        ##############################################################

        return loss, grads
