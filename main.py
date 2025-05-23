"""
Main training script for CNN sign language digit recognition.

This script handles data preprocessing, model training, and evaluation.
It loads the sign language MNIST dataset, trains a CNN model, and evaluates its performance.

Usage:
    python main.py

The script expects CSV files in ./input/train/ and ./input/test/ directories.
"""

import numpy as np
import pandas as pd
import numba as nb
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Import CNN functions using their proper names
from CNN import *

def load_and_preprocess_data():
    """
    Loads and preprocesses the sign language MNIST dataset.
    
    Returns:
        tuple: (X_train, y_train, X_test, y_test) - preprocessed training and testing data
    """
    # Load training and testing datasets
    training_dataframe = pd.read_csv('./input/train/sign_mnist_train.csv')
    testing_dataframe = pd.read_csv('./input/test/sign_mnist_test.csv')
    
    # Filter for digits 0-9 only and separate features from labels
    # Features are in columns 1-784 (pixel values), labels in column 0
    train_mask = training_dataframe['label'] < 10
    test_mask = testing_dataframe['label'] < 10
    
    X_train = training_dataframe[train_mask].values[:, 1:]  # Pixel values
    y_train = training_dataframe[train_mask].values[:, 0]   # Labels
    X_test = testing_dataframe[test_mask].values[:, 1:]     # Pixel values  
    y_test = testing_dataframe[test_mask].values[:, 0]      # Labels
    
    # Normalize pixel values to [0, 1] range
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    # Reshape to proper image format: (num_samples, channels, height, width)
    # Original: (num_samples, 784) -> Target: (num_samples, 1, 28, 28)
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    
    return X_train, y_train, X_test, y_test


def train_cnn_model(X_train, y_train, X_test, y_test):
    """
    Trains the CNN model using the provided training data.
    
    Args:
        X_train (np.ndarray): Training images
        y_train (np.ndarray): Training labels
        X_test (np.ndarray): Testing images (unused in training)
        y_test (np.ndarray): Testing labels (unused in training)
        
    Returns:
        tuple: (training_costs, training_accuracies, trained_parameters, layer_dims, label_template)
    """
    
    # Hyperparameters
    NUM_CLASSES = 10                # Output classes (digits 0-9)
    LEARNING_RATE = 0.0001         # Learning rate for gradient descent
    INPUT_IMAGE_SIZE = 28          # Input image width/height
    INPUT_DEPTH = 1                # Number of input channels (grayscale)
    FILTER_SIZE = 5                # Convolutional filter size (5x5)
    CONV_FILTERS_LAYER1 = 8        # Number of filters in first conv layer
    CONV_FILTERS_LAYER2 = 8        # Number of filters in second conv layer
    BATCH_SIZE = 8                 # Mini-batch size
    MAX_EPOCHS = 3000             # Maximum training epochs
    MOMENTUM_COEFFICIENT = 0.95    # Momentum coefficient for optimization
    
    # Set random seed for reproducibility
    np.random.seed(3424242)

    # Initialize network parameters using Xavier/He initialization
    initialization_scale = 1.0
    weight_std_dev = initialization_scale * np.sqrt(1.0 / (FILTER_SIZE * FILTER_SIZE * INPUT_DEPTH))

    # First convolutional layer parameters
    conv_filters_1 = np.random.normal(
        loc=0, scale=weight_std_dev, 
        size=(CONV_FILTERS_LAYER1, INPUT_DEPTH, FILTER_SIZE, FILTER_SIZE)
    )
    conv_biases_1 = np.zeros((CONV_FILTERS_LAYER1,))
    
    # Second convolutional layer parameters
    conv_filters_2 = np.random.normal(
        loc=0, scale=weight_std_dev, 
        size=(CONV_FILTERS_LAYER2, CONV_FILTERS_LAYER1, FILTER_SIZE, FILTER_SIZE)
    )
    conv_biases_2 = np.zeros((CONV_FILTERS_LAYER2,))

    # Calculate dimensions after convolutions
    conv1_output_size = INPUT_IMAGE_SIZE - FILTER_SIZE + 1  # 28 - 5 + 1 = 24
    conv2_output_size = conv1_output_size - FILTER_SIZE + 1  # 24 - 5 + 1 = 20

    # Fully connected layer parameters
    # After 2x2 max pooling: (20//2) * (20//2) * 8 = 10 * 10 * 8 = 800 features
    fc_input_size = (conv2_output_size // 2) * (conv2_output_size // 2) * CONV_FILTERS_LAYER2
    fc_weights = np.random.rand(NUM_CLASSES, fc_input_size) * 0.01
    fc_biases = np.zeros((NUM_CLASSES, 1))

    # Package all parameters
    network_parameters = (conv_filters_1, conv_filters_2, conv_biases_1, conv_biases_2, fc_weights, fc_biases)
    
    # Layer dimension specifications for efficient computation
    layer_dimensions = (
        INPUT_DEPTH,           # Input channels
        INPUT_IMAGE_SIZE,      # Input size  
        FILTER_SIZE,           # Filter size
        CONV_FILTERS_LAYER1,   # Conv layer 1 channels
        CONV_FILTERS_LAYER2,   # Conv layer 2 channels
        conv1_output_size,     # Conv layer 1 output size
        conv2_output_size      # Conv layer 2 output size
    )
    
    # Training tracking
    training_costs = []
    training_accuracies = []
    
    print("Starting CNN Training...")
    print(f"Architecture: Input({INPUT_IMAGE_SIZE}x{INPUT_IMAGE_SIZE}x{INPUT_DEPTH}) -> Conv1({CONV_FILTERS_LAYER1} filters) -> Conv2({CONV_FILTERS_LAYER2} filters) -> MaxPool -> FC({NUM_CLASSES} classes)")
    print(f"Hyperparameters: LR={LEARNING_RATE}, Batch={BATCH_SIZE}, Epochs={MAX_EPOCHS}, Momentum={MOMENTUM_COEFFICIENT}")
    print("-" * 80)
    
    # Training loop
    for epoch_idx in range(MAX_EPOCHS):
        
        # Unpack current parameters
        conv_filters_1, conv_filters_2, conv_biases_1, conv_biases_2, fc_weights, fc_biases = network_parameters
        
        # Initialize momentum variables
        momentum_vars = initialize_momentum_variables(conv_filters_1, conv_filters_2, conv_biases_1, conv_biases_2, fc_weights, fc_biases)
        (grad_filter1, grad_filter2, grad_fc_weights, grad_bias1, grad_bias2, grad_fc_bias,
         velocity_filter1, velocity_filter2, velocity_fc_weights, velocity_bias1, velocity_bias2, velocity_fc_bias) = momentum_vars
        
        # Shuffle training data for better convergence
        X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train, random_state=0)
        
        # Create mini-batches
        mini_batch_list = create_mini_batches(X_train_shuffled, y_train_shuffled, BATCH_SIZE)
        
        # Randomly select one mini-batch per epoch
        batch_idx = np.random.randint(0, len(mini_batch_list))
        X_batch, y_batch = mini_batch_list[batch_idx]

        # Initialize epoch metrics
        correct_predictions = 0
        epoch_total_cost = 0

        # Process each sample in the mini-batch
        for sample_idx in range(BATCH_SIZE):

            # Create one-hot encoded label
            one_hot_label = np.zeros((fc_weights.shape[0], 1))
            one_hot_label[int(y_batch[sample_idx]), 0] = 1

            # Forward propagation
            forward_outputs = forward(X_batch[sample_idx], network_parameters, layer_dimensions)

            # Compute loss and predictions
            sample_cost, class_probabilities = softmax_cross_entropy_loss(forward_outputs[-1], one_hot_label)
            
            # Check if prediction is correct
            if np.argmax(class_probabilities) == np.argmax(one_hot_label):
                correct_predictions += 1
            epoch_total_cost += sample_cost
                
            # Get max pooling indices for backpropagation
            max_pool_indices = get_max_pool_indices(forward_outputs[1], layer_dimensions)
            
            # Backward propagation
            gradients = backward(X_batch[sample_idx], one_hot_label, network_parameters, 
                               forward_outputs, class_probabilities, max_pool_indices, layer_dimensions)
            grad_f1, grad_f2, grad_b1, grad_b2, grad_fc_w, grad_fc_b = gradients

            # Accumulate gradients
            grad_filter2 += grad_f2
            grad_bias2 += grad_b2
            grad_filter1 += grad_f1
            grad_bias1 += grad_b1
            grad_fc_weights += grad_fc_w
            grad_fc_bias += grad_fc_b
            
        # Store training metrics
        training_accuracies.append(correct_predictions)
        training_costs.append(epoch_total_cost / BATCH_SIZE)

        # Update parameters using momentum-based gradient descent
        # Filter 1 updates
        velocity_filter1 = MOMENTUM_COEFFICIENT * velocity_filter1 - LEARNING_RATE * grad_filter1 / BATCH_SIZE
        conv_filters_1 += velocity_filter1
        velocity_bias1 = MOMENTUM_COEFFICIENT * velocity_bias1 - LEARNING_RATE * grad_bias1 / BATCH_SIZE
        conv_biases_1 += velocity_bias1

        # Filter 2 updates
        velocity_filter2 = MOMENTUM_COEFFICIENT * velocity_filter2 - LEARNING_RATE * grad_filter2 / BATCH_SIZE
        conv_filters_2 += velocity_filter2
        velocity_bias2 = MOMENTUM_COEFFICIENT * velocity_bias2 - LEARNING_RATE * grad_bias2 / BATCH_SIZE
        conv_biases_2 += velocity_bias2

        # Fully connected layer updates
        velocity_fc_weights = MOMENTUM_COEFFICIENT * velocity_fc_weights - LEARNING_RATE * grad_fc_weights / BATCH_SIZE
        fc_weights += velocity_fc_weights
        velocity_fc_bias = MOMENTUM_COEFFICIENT * velocity_fc_bias - LEARNING_RATE * grad_fc_bias / BATCH_SIZE
        fc_biases += velocity_fc_bias

        # Update network parameters
        network_parameters = (conv_filters_1, conv_filters_2, conv_biases_1, conv_biases_2, fc_weights, fc_biases)

        # Print progress every 150 epochs
        if epoch_idx % 150 == 0:
            batch_accuracy = correct_predictions / BATCH_SIZE
            epoch_cost = training_costs[-1][0] if isinstance(training_costs[-1], np.ndarray) else training_costs[-1]
            print(f"Epoch: {epoch_idx:4d} | Batch Accuracy: {batch_accuracy:.3f} | Cost: {epoch_cost:.3f}")
    
    print("-" * 80)
    print("Training completed!")
    
    return training_costs, training_accuracies, network_parameters, layer_dimensions, one_hot_label 


def visualize_training_progress(training_costs, training_accuracies):
    """
    Creates visualization plots for training cost and accuracy over time.
    
    Args:
        training_costs (list): List of training costs per epoch
        training_accuracies (list): List of training accuracies per epoch
    """
    costs_array = np.array(training_costs)
    accuracies_array = np.array(training_accuracies)

    plt.figure(figsize=(12, 5))
    
    # Plot training cost
    plt.subplot(1, 2, 1)
    plt.plot(costs_array, "-b", label="Training Cost", linewidth=2)
    plt.title("Training Cost Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot training accuracy
    plt.subplot(1, 2, 2)
    plt.plot(accuracies_array, "-r", label="Training Accuracy", linewidth=2)
    plt.title("Training Accuracy Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Correct Predictions per Batch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def evaluate_model_accuracy(X_test, y_test, trained_parameters, layer_dimensions, label_template):
    """
    Evaluates the trained model on the test dataset and prints accuracy.
    
    Args:
        X_test (np.ndarray): Test images
        y_test (np.ndarray): Test labels
        trained_parameters (tuple): Trained network parameters
        layer_dimensions (tuple): Layer dimension specifications
        label_template (np.ndarray): Template for one-hot encoding (unused in current implementation)
    """
    print("Evaluating model on test set...")
    
    correct_predictions = 0
    total_samples = len(X_test)
    
    # Process each test sample
    for sample_idx in range(total_samples):
        # Forward propagation
        forward_outputs = forward(X_test[sample_idx], trained_parameters, layer_dimensions)
        
        # Get predictions using softmax
        _, class_probabilities = softmax_cross_entropy_loss(forward_outputs[-1], label_template)
        
        # Check if prediction matches true label
        predicted_class = np.argmax(class_probabilities)
        true_class = y_test[sample_idx]
        
        if predicted_class == true_class:
            correct_predictions += 1
    
    # Calculate and display accuracy
    test_accuracy = (correct_predictions / total_samples) * 100
    print(f"Test Accuracy: {test_accuracy:.2f}% ({correct_predictions}/{total_samples})")
    
    return test_accuracy


def main():
    """
    Main function that orchestrates the entire training and evaluation process.
    
    This function:
    1. Loads and preprocesses the data
    2. Trains the CNN model
    3. Visualizes training progress
    4. Evaluates model performance on test set
    """
    print("=" * 80)
    print("CNN SIGN LANGUAGE DIGIT RECOGNITION")
    print("=" * 80)
    
    # Step 1: Load and preprocess data
    print("Loading and preprocessing data...")
    X_train, y_train, X_test, y_test = load_and_preprocess_data()
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Image shape: {X_train[0].shape}")
    print()
    
    # Step 2: Train the model
    training_results = train_cnn_model(X_train, y_train, X_test, y_test)
    training_costs, training_accuracies, trained_parameters, layer_dimensions, label_template = training_results
    
    # Step 3: Visualize training progress
    print("\nGenerating training progress visualization...")
    visualize_training_progress(training_costs, training_accuracies)
    
    # Step 4: Evaluate on test set
    print()
    final_accuracy = evaluate_model_accuracy(X_test, y_test, trained_parameters, layer_dimensions, label_template)
    
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Final Test Accuracy: {final_accuracy:.2f}%")
    print(f"Total Training Epochs: {len(training_costs)}")
    print(f"Final Training Cost: {training_costs[-1]:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()