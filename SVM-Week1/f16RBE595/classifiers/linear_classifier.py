import numpy as np
from f16RBE595.classifiers.linear_svm import *
from f16RBE595.classifiers.softmax import *

class LinearClassifier:

  def __init__(self):
    self.W = None

  def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this linear classifier using stochastic gradient descent.

    Inputs:
    - X: D x N array of training data. Each training point is a D-dimensional
         column.
    - y: 1-dimensional array of length N with labels 0...K-1, for K classes.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Outputs:
    A list containing the value of the loss function at each training iteration.
    """
    dim, num_train = X.shape
    new_learning_rate = learning_rate
    # print X.dtype
    num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
    # print "Num Classes: ", num_classes
    if self.W is None:
      # lazily initialize W
      self.W = np.random.randn(num_classes, dim) * 0.00001

    # Run stochastic gradient descent to optimize W
    loss_history = []
    batch_data_indices = []
    for it in xrange(num_iters):
      
      #########################################################################
      # TODO:                                                                 #
      # Sample batch_size elements from the training data and their           #
      # corresponding labels to use in this round of gradient descent.        #
      # Store the data in X_batch and their corresponding labels in           #
      # y_batch; after sampling X_batch should have shape (dim, batch_size)   #
      # and y_batch should have shape (batch_size,)                           #
      #                                                                       #
      # Hint: Use np.random.choice to generate indices. Sampling with         #
      # replacement is faster than sampling without replacement.              #
      #########################################################################
      pass
      indices = np.array(range(num_train))
      indices = np.delete(indices, batch_data_indices) 
      batch_data_indices = np.random.choice(indices, batch_size)
      X_batch = X[:, batch_data_indices]
      y_batch = np.take(y, batch_data_indices)
  
      # print "X_batch shape: ", np.shape(X_batch) 
      # print "Y_batch shape: ", np.shape(y_batch) 

      #########################################################################
      #                       END OF YOUR CODE                                #
      #########################################################################

      # evaluate loss and gradient
      loss, grad = self.loss(X_batch, y_batch, reg)
      # loss, grad = softmax_loss_naive(self.W, X_batch, y_batch, reg)
      loss_history.append(loss)

      # perform parameter update
      #########################################################################
      # TODO:                                                                 #
      # Update the weights using the gradient and the learning rate.          #
      #########################################################################
      # new_learning_rate = new_learning_rate/2
      self.W += grad*new_learning_rate
      # self.W[-1, :] = 0.0001
      #########################################################################
      #                       END OF YOUR CODE                                #
      #########################################################################

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

    return loss_history

  def predict(self, X):
    """
    Use the trained weights of this linear classifier to predict labels for
    data points.

    Inputs:
    - X: D x N array of training data. Each column is a D-dimensional point.

    Returns:
    - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
      array of length N, and each element is an integer giving the predicted
      class.
    """
    dim, num_samples = X.shape
    y_pred = np.zeros(X.shape[1])
    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Store the predicted labels in y_pred.            #
    ###########################################################################
    pass

    for i in xrange(num_samples):
      score = np.matmul(self.W, X[:, i])
      y_pred[i] = np.argmax(score)
      # print "Score Matrix Size: ", score.shape, "Predicted Class: ", y_pred

    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################
    return y_pred
  
  def loss(self, X_batch, y_batch, reg):
    """
    Compute the loss function and its derivative. 
    Subclasses will override this.

    Inputs:
    - X_batch: D x N array of data; each column is a data point.
    - y_batch: 1-dimensional array of length N with labels 0...K-1, for K classes.
    - reg: (float) regularization strength.

    Returns: A tuple containing:
    - loss as a single float
    - gradient with respect to self.W; an array of the same shape as W
    """
    num_dim, num_samples = X_batch.shape
    num_classes = self.W.shape[0]
    dW = np.zeros(self.W.shape)

    delta = 1
    loss_image_arr = np.zeros([num_samples, num_classes])

    # Information about various dimensions
    # print "Num Dimensions: ", num_dim, "Num Samples: ", num_samples, "Num Classes: ", num_classes

    score_matrix = np.matmul(self.W, X_batch)
    for i in range(num_samples):

      # Sanity Check for the sizes of he matrices after multiplication
      # All rows in score_matrix represents the score of an image in a class
      # print "Weight Matrix Shape: ", self.W.shape, "Score Matrix Shape: ", score_matrix.shape

      for j in range(num_classes):
        if (j!=y_batch[i]):
          loss_image_arr[i, j] = max(0, score_matrix[j, i] - score_matrix[y_batch[i], i] + delta)**2

    reg_loss  = reg * np.sum(self.W*self.W)
    loss_iter = np.sum(loss_image_arr)/num_samples + reg_loss

    # print "Size of Loss 2D matrix: ",  loss_image_arr

    binary_matrix = loss_image_arr>0
    # print "Binary Matrix: ", binary_matrix.shape, "X_batch: ", X_batch.shape

    dW = np.transpose(np.matmul(X_batch, binary_matrix))
    dW = -dW/num_samples
    # print "Iteration -- ", "Loss: ", loss_iter , "Gradient Shape: ", dW.shape, "Weight Shape: ", self.W.shape

    return loss_iter, dW


class LinearSVM(LinearClassifier):
  """ A subclass that uses the Multiclass SVM loss function """

  # def loss(self, X_batch, y_batch, reg):
  #   return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """

  def loss(self, X_batch, y_batch, reg):
    return softmax_loss_naive(self.W, X_batch, y_batch, reg)
    # return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)

