
import numpy as np
import pandas as pd
import numba as nb
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


from CNN import CNN
from Utils import (
    load_sign_mnist_data, 
    create_batches, 
    evaluate_model, 
    plot_training_progress,
    visualize_predictions,
    print_dataset_info,
)


def trainCNN(trainPath, testPath, config = None):
    # Default configuration
    defaultConfig = {
        'num_output': 10,
        'learning_rate': 0.0001,
        'img_width': 28,
        'img_depth': 1,
        'filter_size': 5,
        'num_filt1': 8,
        'num_filt2': 8,
        'batch_size': 8,
        'num_epochs': 3000,
        'momentum': 0.95,
        'print_every': 150,
        'evaluate_every': 500,
    }

    if config:
        defaultConfig.update(config)
    cfg = defaultConfig
    X_train, y_train, X_test, y_test = load_sign_mnist_data(trainPath, testPath)
    print_dataset_info(X_train, y_train, X_test, y_test)
    cnn = CNN(
        num_output=cfg['num_output'],
        learning_rate=cfg['learning_rate'],
        img_width=cfg['img_width'],
        img_depth=cfg['img_depth'],
        filter_size=cfg['filter_size'],
        num_filt1=cfg['num_filt1'],
        num_filt2=cfg['num_filt2'],
        momentum=cfg['momentum']
    )


    print(f"CNN Architecture:")
    print(f"  Input: {cfg['img_width']}x{cfg['img_width']}x{cfg['img_depth']}")
    print(f"  Conv1: {cfg['num_filt1']} filters of size {cfg['filter_size']}x{cfg['filter_size']}")
    print(f"  Conv2: {cfg['num_filt2']} filters of size {cfg['filter_size']}x{cfg['filter_size']}")
    print(f"  Output: {cfg['num_output']} classes")
    print("="*50)

    # Training tracking
    train_accuracies = []
    train_losses = []
    test_accuracies = []


    for epoch in range(cfg['num_epochs']):
        print(f"Epoch: {epoch}")

        # Shuffle training data
        X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train, random_state=epoch)

        print("1")        
        
        # Create batches
        batches = create_batches(X_train_shuffled, y_train_shuffled, cfg['batch_size'])
        print("2")        
        
        epoch_accuracy = 0
        epoch_loss = 0
        
        # Train on all batches
        for batch_idx, (X_batch, y_batch) in enumerate(batches):
            print(f"train {batch_idx}")
            batch_acc, batch_loss = cnn.train_batch(X_batch, y_batch)
            epoch_accuracy += batch_acc
            epoch_loss += batch_loss
        
        print("3")        
        # Average over all batches
        epoch_accuracy /= len(batches)
        epoch_loss /= len(batches)
        
        train_accuracies.append(epoch_accuracy)
        train_losses.append(epoch_loss)
        
        # Print progress
        if epoch % cfg['print_every'] == 0:
            print(f"Epoch {epoch:4d}: Accuracy={epoch_accuracy:.3f}, Loss={epoch_loss:.3f}")
        
        # Evaluate on test set periodically
        if epoch % cfg['evaluate_every'] == 0 and epoch > 0:
            test_acc = evaluate_model(cnn, X_test, y_test, batch_size=32)
            test_accuracies.append(test_acc)
            print(f"         Test Accuracy: {test_acc:.3f}")

    # Final evaluation
    print("Evaluating on test set...")
    final_test_acc = evaluate_model(cnn, X_test, y_test, batch_size=32)
    print(f"Final Test Accuracy: {final_test_acc:.3f}")


    # Plot training progress
    print("Plotting training progress...")
    plot_training_progress(train_accuracies, train_losses)


    # Visualize some predictions
    print("Visualizing predictions...")
    visualize_predictions(cnn, X_test, y_test, num_samples=8)

    return cnn, train_accuracies, train_losses, test_accuracies


def main():
    TRAIN_PATH = './input/train/sign_mnist_train.csv'
    TEST_PATH = './input/test/sign_mnist_test.csv'

    cnn, _, _, _ = trainCNN(TRAIN_PATH, TEST_PATH)


if __name__ == "__main__":
    main()

