import numpy as np
import pandas as pd
import numba as nb



@nb.jit(nopython=True)
def _initGrads(sizes):
    """Initialize gradient arrays."""
    l, w, f, l1, l2, w1, w2 = sizes
    
    dconv2 = np.zeros((l2, w2, w2))
    dconv1 = np.zeros((l1, w1, w1))
    
    dfilt2 = np.zeros((l2, l1, f, f))
    dbias2 = np.zeros((l2,))
    
    dfilt1 = np.zeros((l1, l, f, f))
    dbias1 = np.zeros((l1,))
    
    return dconv1, dconv2, dfilt1, dfilt2, dbias1, dbias2

@nb.jit(nopython=True)
def _backward(image, label, filt1, filt2, theta3, delta, probs, nanarg, sizes):
    """Backward propagation to compute gradients."""
    l, w, f, l1, l2, w1, w2 = sizes
    conv1, conv2, pooled_layer, fc1, out = delta
    dconv1, dconv2, dfilt1, dfilt2, dbias1, dbias2 = _initGrads(sizes)
    
    # Output layer gradients
    dout = probs - label
    dtheta3 = dout.dot(fc1.T)
    dbias3 = dout
    
    # Fully connected layer gradients
    dfc1 = theta3.T.dot(dout)
    dpool = dfc1.T.reshape((l2, w2//2, w2//2))
    
    # Max pooling backward pass
    for jj in range(0, l2):
        for i in range(0, w2, 2):
            for j in range(0, w2, 2):
                a = int(nanarg[jj, i, j, 0])
                b = int(nanarg[jj, i, j, 1])
                dconv2[jj, i+a, j+b] = dpool[jj, i//2, j//2]
    
    dconv2 = dconv2 * _drelu(conv2)
    
    # Second convolutional layer gradients
    for jj in range(0, l2):
        for x in range(0, w2):
            for y in range(0, w2):
                dfilt2[jj] += dconv2[jj, x, y] * conv1[:, x:x+f, y:y+f]
                dconv1[:, x:x+f, y:y+f] += dconv2[jj, x, y] * filt2[jj]
        dbias2[jj] = np.sum(dconv2[jj])
    
    dconv1 = dconv1 * _drelu(conv1)
    
    # First convolutional layer gradients
    for jj in range(0, l1):
        for x in range(0, w1):
            for y in range(0, w1):
                dfilt1[jj] += dconv1[jj, x, y] * image[:, x:x+f, y:y+f]
        dbias1[jj] = np.sum(dconv1[jj])
    
    return dfilt1, dfilt2, dbias1, dbias2, dtheta3, dbias3


@nb.jit(nopython=True)
def _maxPool(X, f, s):
    """Max pooling operation."""
    (l, w, w) = X.shape
    pool = np.zeros((l, (w-f)//s+1, (w-f)//s+1))
    for jj in range(0, l):
        for i in range(0, w, s):
            for j in range(0, w, s):
                pool[jj, i//2, j//2] = np.max(X[jj, i:i+f, j:j+f])
    return pool

@nb.jit(nopython=True)
def _relu(x):
    """ReLU activation function."""
    return x * (x > 0)

@nb.jit(nopython=True)
def _drelu(x):
    """Derivative of ReLU activation function."""
    return 1. * (x > 0)


@nb.jit(nopython=True)
def _forward(image, filt1, filt2, bias1, bias2, theta3, bias3, sizes):
    """Forward propagation through the network."""
    l, w, f, l1, l2, w1, w2 = sizes
    
    conv1 = np.zeros((l1, w1, w1))
    conv2 = np.zeros((l2, w2, w2))
    
    # First convolutional layer
    for jj in range(0, l1):
        for x in range(0, w1):
            for y in range(0, w1):
                conv1[jj, x, y] = np.sum(image[:, x:x+f, y:y+f] * filt1[jj]) + bias1[jj]
    conv1 = _relu(conv1)
    
    # Second convolutional layer
    for jj in range(0, l2):
        for x in range(0, w2):
            for y in range(0, w2):
                conv2[jj, x, y] = np.sum(conv1[:, x:x+f, y:y+f] * filt2[jj]) + bias2[jj]
    conv2 = _relu(conv2)
    
    # Max pooling
    pooled_layer = _maxPool(conv2, 2, 2)
    
    # Fully connected layer
    fc1 = pooled_layer.reshape(((w2//2) * (w2//2) * l2, 1))
    out = theta3.dot(fc1) + bias3
    
    return conv1, conv2, pooled_layer, fc1, out


class CNN:
    def __init__(self, num_output=10, learning_rate=0.0001, img_width=28, img_depth=1, 
                 filter_size=5, num_filt1=8, num_filt2=8, momentum=0.95):
        """
        Initialize CNN with specified architecture parameters.
        """
        self.num_output = num_output
        self.learning_rate = learning_rate
        self.img_width = img_width
        self.img_depth = img_depth
        self.filter_size = filter_size
        self.num_filt1 = num_filt1
        self.num_filt2 = num_filt2
        self.momentum = momentum
        
        # Calculate dimensions
        self.w1 = img_width - filter_size + 1  # After first conv
        self.w2 = self.w1 - filter_size + 1    # After second conv
        
        # Initialize network parameters
        self._initializeParameters()
        
        # Initialize momentum terms
        self._initializeMomentum()
        
        # Set up sizes tuple for numba functions
        self.sizes = (img_depth, img_width, filter_size, num_filt1, num_filt2, self.w1, self.w2)

    def _initializeParameters(self):
        """Initialize network weights and biases."""
        np.random.seed(3424242)
        
        # Xavier initialization
        scale = 1.0
        stddev = scale * np.sqrt(1. / (self.filter_size * self.filter_size * self.img_depth))
        
        # Convolutional layers
        self.filt1 = np.random.normal(loc=0, scale=stddev, 
                                     size=(self.num_filt1, self.img_depth, self.filter_size, self.filter_size))
        self.bias1 = np.zeros((self.num_filt1,))
        
        self.filt2 = np.random.normal(loc=0, scale=stddev, 
                                     size=(self.num_filt2, self.num_filt1, self.filter_size, self.filter_size))
        self.bias2 = np.zeros((self.num_filt2,))
        
        # Fully connected layer
        fc_input_size = (self.w2 // 2) * (self.w2 // 2) * self.num_filt2
        self.theta3 = np.random.rand(self.num_output, fc_input_size) * 0.01
        self.bias3 = np.zeros((self.num_output, 1))

    def _initializeMomentum(self):
        """Initialize momentum terms for all parameters."""
        self.v1 = np.zeros_like(self.filt1)
        self.bv1 = np.zeros_like(self.bias1)
        
        self.v2 = np.zeros_like(self.filt2)
        self.bv2 = np.zeros_like(self.bias2)
        
        self.v3 = np.zeros_like(self.theta3)
        self.bv3 = np.zeros_like(self.bias3)

    def _softmaxLoss(self, out, y):
        """Compute softmax probabilities and cross-entropy loss."""
        eout = np.exp(out, dtype=np.float64)
        probs = eout / sum(eout)
        
        p = sum(y * probs)
        loss = -np.log(p)
        return loss, probs
    
    def _getNaNargmax(self, conv2):
        """Get indices of maximum values for max pooling backward pass."""
        def nanargmax(a):
            idx = np.argmax(a, axis=None)
            multi_idx = np.unravel_index(idx, a.shape)
            if np.isnan(a[multi_idx]):
                nan_count = np.sum(np.isnan(a))
                idx = np.argpartition(a, -nan_count-1, axis=None)[-nan_count-1]
                multi_idx = np.unravel_index(idx, a.shape)
            return multi_idx
        
        l, w, f, l1, l2, w1, w2 = self.sizes
        nanarg = np.zeros((l2, w2, w2, 2))
        
        for jj in range(0, l2):
            for i in range(0, w2, 2):
                for j in range(0, w2, 2):
                    (a, b) = nanargmax(conv2[jj, i:i+2, j:j+2])
                    nanarg[jj, i, j, 0] = a
                    nanarg[jj, i, j, 1] = b
        
        return nanarg

    def forward(self, image):
        """Forward pass through the network."""
        return _forward(image, self.filt1, self.filt2, self.bias1, self.bias2, 
                       self.theta3, self.bias3, self.sizes)
    
    def predict(self, image):
        """Make a prediction for a single image."""
        _, _, _, _, out = self.forward(image)
        # Use a dummy label for softmax calculation
        dummy_label = np.ones_like(out) / len(out)
        _, probs = self._softmaxLoss(out, dummy_label)
        return np.argmax(probs)

    def train_batch(self, X_batch, y_batch):
        """Train on a single batch of data."""
        batch_size = len(X_batch)
        
        # Initialize gradient accumulators
        dfilt1_total = np.zeros_like(self.filt1)
        dfilt2_total = np.zeros_like(self.filt2)
        dbias1_total = np.zeros_like(self.bias1)
        dbias2_total = np.zeros_like(self.bias2)
        dtheta3_total = np.zeros_like(self.theta3)
        dbias3_total = np.zeros_like(self.bias3)
        
        total_cost = 0
        correct_predictions = 0
        
        for i in range(batch_size):
            # Create one-hot label
            label = np.zeros((self.num_output, 1))
            label[int(y_batch[i]), 0] = 1
            
            # Forward pass
            delta = self.forward(X_batch[i])
            
            # Compute cost and accuracy
            cost, probs = self._softmaxLoss(delta[-1], label)
            total_cost += cost
            
            if np.argmax(probs) == np.argmax(label):
                correct_predictions += 1
            
            # Backward pass
            nanarg = self._getNaNargmax(delta[1])
            dfilt1, dfilt2, dbias1, dbias2, dtheta3, dbias3 = _backward(
                X_batch[i], label, self.filt1, self.filt2, self.theta3, 
                delta, probs, nanarg, self.sizes
            )
            
            # Accumulate gradients
            dfilt1_total += dfilt1
            dfilt2_total += dfilt2
            dbias1_total += dbias1
            dbias2_total += dbias2
            dtheta3_total += dtheta3
            dbias3_total += dbias3
        
        # Update parameters with momentum
        self.v1 = self.momentum * self.v1 - self.learning_rate * dfilt1_total / batch_size
        self.filt1 += self.v1
        self.bv1 = self.momentum * self.bv1 - self.learning_rate * dbias1_total / batch_size
        self.bias1 += self.bv1
        
        self.v2 = self.momentum * self.v2 - self.learning_rate * dfilt2_total / batch_size
        self.filt2 += self.v2
        self.bv2 = self.momentum * self.bv2 - self.learning_rate * dbias2_total / batch_size
        self.bias2 += self.bv2
        
        self.v3 = self.momentum * self.v3 - self.learning_rate * dtheta3_total / batch_size
        self.theta3 += self.v3
        self.bv3 = self.momentum * self.bv3 - self.learning_rate * dbias3_total / batch_size
        self.bias3 += self.bv3
        
        return correct_predictions / batch_size, total_cost / batch_size