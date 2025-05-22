import numpy as np
import pandas as pd
import numba as nb
import matplotlib as plt


def load_sign_mnist_data(train_path, test_path):
    """
    Load and preprocess Sign Language MNIST dataset.
    
    Args:
        train_path (str): Path to training CSV file
        test_path (str): Path to test CSV file
    
    Returns:
        tuple: (X_train, y_train, X_test, y_test) preprocessed data
    """
    # Load data
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # Filter to only include digits 0-9 (exclude letters J and Z which are 9 and 25)
    X_train = df_train[df_train['label'] < 10].values[:, 1:]
    y_train = df_train[df_train['label'] < 10].values[:, 0]
    
    X_test = df_test[df_test['label'] < 10].values[:, 1:]
    y_test = df_test[df_test['label'] < 10].values[:, 0]
    
    # Normalize pixel values to [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    # Reshape to format (num_samples, 1, 28, 28)
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    
    return X_train, y_train, X_test, y_test

def create_batches(X, y, batch_size):
    """
    Create batches from the dataset.
    
    Args:
        X (np.array): Input data
        y (np.array): Labels
        batch_size (int): Size of each batch
    
    Returns:
        list: List of (X_batch, y_batch) tuples
    """
    batches = []
    
    for idx in np.array_split(np.arange(len(X)), len(X) // batch_size):
        if len(idx) == batch_size:  # Only include full batches
            X_batch = X[idx]
            y_batch = y[idx]
            batches.append((X_batch, y_batch))
    
    return batches


def evaluate_model(cnn, X_test, y_test, batch_size=32):
    """
    Evaluate the CNN model on test data.
    
    Args:
        cnn: Trained CNN model
        X_test (np.array): Test images
        y_test (np.array): Test labels
        batch_size (int): Batch size for evaluation
    
    Returns:
        float: Test accuracy
    """
    correct_predictions = 0
    total_samples = 0
    
    # Create test batches
    test_batches = create_batches(X_test, y_test, batch_size)
    
    for X_batch, y_batch in test_batches:
        for i in range(len(X_batch)):
            prediction = cnn.predict(X_batch[i])
            if prediction == y_batch[i]:
                correct_predictions += 1
            total_samples += 1
    
    return correct_predictions / total_samples


def plot_training_progress(accuracies, losses, save_path=None):
    """
    Plot training accuracy and loss curves.
    
    Args:
        accuracies (list): Training accuracies over epochs
        losses (list): Training losses over epochs
        save_path (str, optional): Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(accuracies)
    ax1.set_title('Training Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(losses)
    ax2.set_title('Training Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def visualize_predictions(cnn, X_test, y_test, num_samples=8):
    """
    Visualize model predictions on test samples.
    
    Args:
        cnn: Trained CNN model
        X_test (np.array): Test images
        y_test (np.array): Test labels
        num_samples (int): Number of samples to visualize
    """
    # Select random samples
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        # Make prediction
        prediction = cnn.predict(X_test[idx])
        actual = y_test[idx]
        
        # Plot image
        axes[i].imshow(X_test[idx].reshape(28, 28), cmap='gray')
        axes[i].set_title(f'Pred: {prediction}, Actual: {actual}')
        axes[i].axis('off')
        
        # Color title based on correctness
        if prediction == actual:
            axes[i].title.set_color('green')
        else:
            axes[i].title.set_color('red')
    
    plt.tight_layout()
    plt.show()


def print_dataset_info(X_train, y_train, X_test, y_test):
    """
    Print information about the dataset.
    
    Args:
        X_train, y_train, X_test, y_test: Dataset arrays
    """
    print("Dataset Information:")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Image shape: {X_train[0].shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    print(f"Class distribution (train): {np.bincount(y_train.astype(int))}")
    print(f"Class distribution (test): {np.bincount(y_test.astype(int))}")


# get the highest (index) value of the arrays
def get_nanargmax(conv2, sizes):

    def nanargmax(a):
        idx = np.argmax(a, axis=None)
        multi_idx = np.unravel_index(idx, a.shape)
        if np.isnan(a[multi_idx]):
            nan_count = np.sum(np.isnan(a))
            idx = np.argpartition(a, -nan_count-1, axis=None)[-nan_count-1]
            multi_idx = np.unravel_index(idx, a.shape)
        return multi_idx

    l, w, f, l1, l2, w1, w2 = sizes
    nanarg = np.zeros((l2, w2, w2, 2))

    for jj in range(0, l2):
        for i in range(0, w2, 2):
            for j in range(0, w2, 2):
                (a,b) = nanargmax(conv2[jj,i:i+2,j:j+2])
                nanarg[jj, i, j, 0] = a
                nanarg[jj, i, j, 1] = b

    return nanarg