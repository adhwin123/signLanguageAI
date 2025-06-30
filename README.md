

#  Sign Language Recognition with CNNs

This project builds and trains a Convolutional Neural Network (CNN) to classify American Sign Language (ASL) gestures using the MNIST Sign Language dataset. It is part of coursework for **Artificial Intelligence** at **Columbia University** under **Professor Ansaf Sales-Aouissi**.

---

##  Project Overview

* **Course**: Artificial Intelligence
* **Instructor**: Prof. Ansaf Sales-Aouissi
* **Institution**: Columbia University
* **Files**:

  * `sign_language.py`: Model class implementation
  * `sign_language.ipynb`: Jupyter notebook for data loading, training, and evaluation

---

##  Core Functionality

### `sign_language.py`

Defines a class `SignLanguage` encapsulating model creation, compilation, training, and evaluation:

* `create_model()`: Constructs a CNN with:

  * 2 Convolutional layers
  * MaxPooling
  * Dropout for regularization
  * Dense softmax output layer

* `train_model(x_train, y_train, x_val, y_val, epochs, batch_size)`: Trains the CNN.

* `evaluate(x_test, y_test)`: Outputs test accuracy.

* `predict(x)`: Predicts labels for new samples.

### `sign_language.ipynb`

* Loads and preprocesses the MNIST Sign Language dataset.
* Normalizes and reshapes input data to `(28, 28, 1)` for grayscale image input.
* One-hot encodes target labels.
* Trains the CNN using the class in `sign_language.py`.
* Plots training history and evaluates performance.
* Optionally includes UNI placeholder for submission.

---

##  Technologies Used

* Python
* NumPy, Pandas, Matplotlib
* TensorFlow, Keras
* scikit-learn

---

##  Dataset

You can obtain the Sign Language MNIST dataset from:

* [Kaggle: Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)

Format:

* 28x28 grayscale images of hand gestures.
* CSV files: `sign_mnist_train.csv` and `sign_mnist_test.csv`.

---

##  How to Run

1. **Install dependencies**:

   ```bash
   pip install numpy pandas matplotlib tensorflow scikit-learn keras
   ```

2. **Train and evaluate** (via notebook or script):

   ```bash
   python sign_language.py
   ```

3. Or open and run cells in `sign_language.ipynb` for interactive results and plots.

---

##  Results

The trained CNN achieves high accuracy on the test set, depending on number of epochs and dropout rate. Performance can be further improved via:

* Hyperparameter tuning
* Data augmentation
* More complex architectures

---

##  Acknowledgments

* Professor **Ansaf Sales-Aouissi**
* **Columbia University** Department of Computer Science
* Dataset by **Nicholas Renotte** and contributors on **Kaggle**

