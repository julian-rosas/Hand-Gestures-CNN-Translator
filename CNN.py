"""
Convolutional Neural Network Implementation for Sign Language Digit Recognition

This module implements a CNN from scratch using NumPy and Numba for performance optimization.
The network architecture consists of two convolutional layers followed by max pooling and 
a fully connected output layer for classifying sign language digits (0-9).

Network Architecture:
- Input: 28x28x1 grayscale images
- Conv Layer 1: 8 filters of size 5x5 with ReLU activation
- Conv Layer 2: 8 filters of size 5x5 with ReLU activation  
- Max Pooling: 2x2 with stride 2
- Fully Connected: Output layer with softmax for 10 classes

Author: Julian Rosas Scull
Date: 29 May 2025
"""

import numpy as np
import pandas as pd
import numba as nb
from sklearn.utils import shuffle


@nb.jit(nopython=True)
def max_pool_2d(feature_maps, pool_size, stride):
    """
    Performs 2D max pooling operation on feature maps.
    
    Args:
        feature_maps (np.ndarray): Input feature maps of shape (num_channels, height, width)
        pool_size (int): Size of the pooling window (assumes square window)
        stride (int): Stride for the pooling operation
        
    Returns:
        np.ndarray: Pooled feature maps of shape (num_channels, pooled_height, pooled_width)
    """
    num_channels, height, width = feature_maps.shape
    pooled_height = (height - pool_size) // stride + 1
    pooled_width = (width - pool_size) // stride + 1
    
    pooled_output = np.zeros((num_channels, pooled_height, pooled_width))
    
    for channel_idx in range(num_channels):
        for row_idx in range(0, height, stride):
            for col_idx in range(0, width, stride):
                pooled_output[channel_idx, row_idx//stride, col_idx//stride] = np.max(
                    feature_maps[channel_idx, row_idx:row_idx+pool_size, col_idx:col_idx+pool_size]
                )
    return pooled_output


@nb.jit(nopython=True)
def relu_activation(input_tensor):
    """
    ReLU (Rectified Linear Unit) activation function.
    
    Args:
        input_tensor (np.ndarray): Input tensor
        
    Returns:
        np.ndarray: Output tensor with ReLU applied element-wise
    """
    return input_tensor * (input_tensor > 0)


@nb.jit(nopython=True)
def relu_derivative(input_tensor):
    """
    Derivative of ReLU activation function.
    
    Args:
        input_tensor (np.ndarray): Input tensor
        
    Returns:
        np.ndarray: Derivative values (1 for positive inputs, 0 for negative)
    """
    return 1.0 * (input_tensor > 0)


def softmax_cross_entropy_loss(network_output, true_label):
    """
    Computes softmax probabilities and cross-entropy loss.
    
    Args:
        network_output (np.ndarray): Raw network output logits
        true_label (np.ndarray): One-hot encoded true label
        
    Returns:
        tuple: (cross_entropy_loss, softmax_probabilities)
    """
    # Use float64 for numerical stability
    exponentials = np.exp(network_output, dtype=np.float64)
    probabilities = exponentials / np.sum(exponentials)
    
    # Cross-entropy loss: -log(probability of correct class)
    correct_class_probability = np.sum(true_label * probabilities)
    cross_entropy_loss = -np.log(correct_class_probability)
    
    return cross_entropy_loss, probabilities


@nb.jit(nopython=True)
def forward(input_image, network_parameters, layer_dimensions):
    """
    Performs forward propagation through the CNN.
    
    Args:
        input_image (np.ndarray): Input image of shape (channels, height, width)
        network_parameters (tuple): Network weights and biases
        layer_dimensions (tuple): Dimensions for each layer
        
    Returns:
        tuple: Intermediate activations and final output
    """
    conv_filter_1, conv_filter_2, conv_bias_1, conv_bias_2, fc_weights, fc_bias = network_parameters
    input_channels, input_size, filter_size, conv1_channels, conv2_channels, conv1_size, conv2_size = layer_dimensions

    # First convolutional layer
    conv1_output = np.zeros((conv1_channels, conv1_size, conv1_size))
    conv2_output = np.zeros((conv2_channels, conv2_size, conv2_size))

    # Convolution operation for first layer
    for filter_idx in range(conv1_channels):
        for row_idx in range(conv1_size):
            for col_idx in range(conv1_size):
                conv1_output[filter_idx, row_idx, col_idx] = (
                    np.sum(input_image[:, row_idx:row_idx+filter_size, col_idx:col_idx+filter_size] * 
                           conv_filter_1[filter_idx]) + conv_bias_1[filter_idx]
                )
    conv1_output = relu_activation(conv1_output)

    # Second convolutional layer
    for filter_idx in range(conv2_channels):
        for row_idx in range(conv2_size):
            for col_idx in range(conv2_size):
                conv2_output[filter_idx, row_idx, col_idx] = (
                    np.sum(conv1_output[:, row_idx:row_idx+filter_size, col_idx:col_idx+filter_size] * 
                           conv_filter_2[filter_idx]) + conv_bias_2[filter_idx]
                )
    conv2_output = relu_activation(conv2_output)

    # Max pooling with 2x2 kernel and stride 2
    pooled_features = max_pool_2d(conv2_output, 2, 2)	

    # Flatten for fully connected layer
    flattened_features = pooled_features.reshape(((conv2_size//2) * (conv2_size//2) * conv2_channels, 1))

    # Fully connected layer output
    final_output = fc_weights.dot(flattened_features) + fc_bias

    return conv1_output, conv2_output, pooled_features, flattened_features, final_output


def get_max_pool_indices(conv2_features, layer_dimensions):
    """
    Gets the indices of maximum values for max pooling operation (needed for backpropagation).
    
    Args:
        conv2_features (np.ndarray): Feature maps from second convolutional layer
        layer_dimensions (tuple): Layer dimension specifications
        
    Returns:
        np.ndarray: Indices of maximum values for each pooling window
    """
    def get_nanargmax(feature_window):
        """Helper function to get argmax while handling NaN values."""
        max_idx = np.argmax(feature_window, axis=None)
        max_indices = np.unravel_index(max_idx, feature_window.shape)
        if np.isnan(feature_window[max_indices]):
            nan_count = np.sum(np.isnan(feature_window))
            max_idx = np.argpartition(feature_window, -nan_count-1, axis=None)[-nan_count-1]
            max_indices = np.unravel_index(max_idx, feature_window.shape)
        return max_indices

    input_channels, input_size, filter_size, conv1_channels, conv2_channels, conv1_size, conv2_size = layer_dimensions
    max_indices = np.zeros((conv2_channels, conv2_size, conv2_size, 2))

    for channel_idx in range(conv2_channels):
        for row_idx in range(0, conv2_size, 2):
            for col_idx in range(0, conv2_size, 2):
                max_row, max_col = get_nanargmax(conv2_features[channel_idx, row_idx:row_idx+2, col_idx:col_idx+2])
                max_indices[channel_idx, row_idx, col_idx, 0] = max_row
                max_indices[channel_idx, row_idx, col_idx, 1] = max_col

    return max_indices


@nb.jit(nopython=True)
def initialize_gradients(layer_dimensions):
    """
    Initializes gradient arrays for backpropagation.
    
    Args:
        layer_dimensions (tuple): Dimensions for each layer
        
    Returns:
        tuple: Initialized gradient arrays
    """
    input_channels, input_size, filter_size, conv1_channels, conv2_channels, conv1_size, conv2_size = layer_dimensions
    
    grad_conv2 = np.zeros((conv2_channels, conv2_size, conv2_size))
    grad_conv1 = np.zeros((conv1_channels, conv1_size, conv1_size))

    grad_filter2 = np.zeros((conv2_channels, conv1_channels, filter_size, filter_size))
    grad_bias2 = np.zeros((conv2_channels,))

    grad_filter1 = np.zeros((conv1_channels, input_channels, filter_size, filter_size))
    grad_bias1 = np.zeros((conv1_channels,))
    
    return grad_conv1, grad_conv2, grad_filter1, grad_filter2, grad_bias1, grad_bias2


@nb.jit(nopython=True)
def backward(input_image, true_label, network_parameters, forward_activations, 
                        softmax_probabilities, max_pool_indices, layer_dimensions):
    """
    Performs backward propagation to compute gradients.
    
    Args:
        input_image (np.ndarray): Original input image
        true_label (np.ndarray): One-hot encoded true label
        network_parameters (tuple): Current network parameters
        forward_activations (tuple): Activations from forward pass
        softmax_probabilities (np.ndarray): Output probabilities from softmax
        max_pool_indices (np.ndarray): Indices from max pooling operation
        layer_dimensions (tuple): Layer dimension specifications
        
    Returns:
        tuple: Gradients for all network parameters
    """
    conv_filter_1, conv_filter_2, conv_bias_1, conv_bias_2, fc_weights, fc_bias = network_parameters
    input_channels, input_size, filter_size, conv1_channels, conv2_channels, conv1_size, conv2_size = layer_dimensions
    conv1_output, conv2_output, pooled_features, flattened_features, final_output = forward_activations
    
    grad_conv1, grad_conv2, grad_filter1, grad_filter2, grad_bias1, grad_bias2 = initialize_gradients(layer_dimensions)

    # Output layer gradients (softmax cross-entropy derivative)
    output_error = softmax_probabilities - true_label
    grad_fc_weights = output_error.dot(flattened_features.T)
    grad_fc_bias = output_error

    # Backpropagate to flattened features
    grad_flattened = fc_weights.T.dot(output_error)
    grad_pooled = grad_flattened.T.reshape((conv2_channels, conv2_size//2, conv2_size//2))

    # Backpropagate through max pooling
    for channel_idx in range(conv2_channels):
        for row_idx in range(0, conv2_size, 2):
            for col_idx in range(0, conv2_size, 2):
                max_row = int(max_pool_indices[channel_idx, row_idx, col_idx, 0])
                max_col = int(max_pool_indices[channel_idx, row_idx, col_idx, 1])
                grad_conv2[channel_idx, row_idx+max_row, col_idx+max_col] = grad_pooled[channel_idx, row_idx//2, col_idx//2]

    # Apply ReLU derivative
    grad_conv2 = grad_conv2 * relu_derivative(conv2_output)

    # Backpropagate through second convolutional layer
    for filter_idx in range(conv2_channels):
        for row_idx in range(conv2_size):
            for col_idx in range(conv2_size):
                grad_filter2[filter_idx] += grad_conv2[filter_idx, row_idx, col_idx] * conv1_output[:, row_idx:row_idx+filter_size, col_idx:col_idx+filter_size]
                grad_conv1[:, row_idx:row_idx+filter_size, col_idx:col_idx+filter_size] += grad_conv2[filter_idx, row_idx, col_idx] * conv_filter_2[filter_idx]
        grad_bias2[filter_idx] = np.sum(grad_conv2[filter_idx])

    # Apply ReLU derivative for first layer
    grad_conv1 = grad_conv1 * relu_derivative(conv1_output)

    # Backpropagate through first convolutional layer
    for filter_idx in range(conv1_channels):
        for row_idx in range(conv1_size):
            for col_idx in range(conv1_size):
                grad_filter1[filter_idx] += grad_conv1[filter_idx, row_idx, col_idx] * input_image[:, row_idx:row_idx+filter_size, col_idx:col_idx+filter_size]
        grad_bias1[filter_idx] = np.sum(grad_conv1[filter_idx])

    return grad_filter1, grad_filter2, grad_bias1, grad_bias2, grad_fc_weights, grad_fc_bias


def initialize_momentum_variables(conv_filter_1, conv_filter_2, conv_bias_1, conv_bias_2, fc_weights, fc_bias):
    """
    Initializes momentum variables for gradient descent with momentum.
    
    Args:
        conv_filter_1, conv_filter_2: Convolutional filters
        conv_bias_1, conv_bias_2: Convolutional biases  
        fc_weights, fc_bias: Fully connected layer parameters
        
    Returns:
        tuple: Gradient and velocity arrays for momentum optimization
    """
    # Initialize gradient arrays
    grad_filter1 = np.zeros((len(conv_filter_1), conv_filter_1[0].shape[0], conv_filter_1[0].shape[1], conv_filter_1[0].shape[2]))
    grad_bias1 = np.zeros((len(conv_filter_1),))
    velocity_filter1 = np.zeros((len(conv_filter_1), conv_filter_1[0].shape[0], conv_filter_1[0].shape[1], conv_filter_1[0].shape[2]))
    velocity_bias1 = np.zeros((len(conv_filter_1),))

    grad_filter2 = np.zeros((len(conv_filter_2), conv_filter_2[0].shape[0], conv_filter_2[0].shape[1], conv_filter_2[0].shape[2]))
    grad_bias2 = np.zeros((len(conv_filter_2),))
    velocity_filter2 = np.zeros((len(conv_filter_2), conv_filter_2[0].shape[0], conv_filter_2[0].shape[1], conv_filter_2[0].shape[2]))
    velocity_bias2 = np.zeros((len(conv_filter_2),))

    grad_fc_weights = np.zeros(fc_weights.shape)
    grad_fc_bias = np.zeros(fc_bias.shape)
    velocity_fc_weights = np.zeros(fc_weights.shape)
    velocity_fc_bias = np.zeros(fc_bias.shape)

    return (grad_filter1, grad_filter2, grad_fc_weights, grad_bias1, grad_bias2, grad_fc_bias, 
            velocity_filter1, velocity_filter2, velocity_fc_weights, velocity_bias1, velocity_bias2, velocity_fc_bias)


def create_mini_batches(input_data, labels, batch_size):
    """
    Creates mini-batches from the input data and labels.
    
    Args:
        input_data (np.ndarray): Input training data
        labels (np.ndarray): Training labels
        batch_size (int): Size of each mini-batch
        
    Returns:
        list: List of (data_batch, label_batch) tuples
    """
    mini_batch_list = []
    
    for batch_indices in np.array_split(np.arange(len(input_data)), len(input_data) / batch_size):
        data_batch = input_data[batch_indices]
        label_batch = labels[batch_indices]
        mini_batch_list.append((data_batch, label_batch))
    
    return mini_batch_list