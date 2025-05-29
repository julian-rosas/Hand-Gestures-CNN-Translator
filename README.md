# Hand-Gestures-CNN-Translator


This project implements a **Convolutional Neural Network (CNN)** from scratch using **NumPy** and **Numba** to recognize hand signs representing digits (0–9) from the **Sign Language MNIST** dataset.

---

## 📂 Project Structure


```
Hand-Gestures-CNN-Translator
│   README.md
│   requirements.txt    
│   main.py   
│   CNN.py
│   SignLanguageCNN_Project.ipynb
│
└───input
    └───train
    │   │   sign_mnist_train.csv
    │
    └───test
        │   sign_mnist_test.csv
```



## 🚀 Usage

### ✅ Option 1: Using Jupyter Notebook (Recommended for Exploration)

1. Look at 📓 [Getting Started Notebook](GettingStarted.ipynb) for a detailed explanation.

### ✅ Option 2: Using the Terminal (Headless Mode)

1. Make sure dependencies are installed:

    ```bash
    pip install -r requirements.txt
    ```

2. Ensure dataset files are placed in the following paths:

    ```
    ./input/train/sign_mnist_train.csv
    ./input/test/sign_mnist_test.csv
    ```

3. Run the training and evaluation:

    ```bash
    python main.py
    ```


