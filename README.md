# mytorch: Custom PyTorch Implementation

This repository is a project for my **Computational Intelligence course**, showcasing a custom-built deep learning framework, **mytorch**, alongside its comparison with native PyTorch implementations on the Fashion MNIST dataset. The project includes implementations of both Convolutional Neural Networks (CNNs) and Multi-Layer Perceptrons (MLPs).

---

## Contents

1. **Notebooks Overview**
    - `Conv_MNIST_mytorch.ipynb`: Implements a Convolutional Neural Network (CNN) on the Fashion MNIST dataset using the custom `mytorch` library.
    - `Conv_MNIST_pytorch.ipynb`: Implements the same CNN using PyTorch for comparison.
    - `MLP_MNIST_pytorch.ipynb`: Implements a Multi-Layer Perceptron (MLP) on the Fashion MNIST dataset using PyTorch.

2. **Key Features**
    - A fully functional custom deep learning framework (`mytorch`), demonstrating an understanding of foundational concepts behind PyTorch.
    - Direct comparisons between the performance and usability of `mytorch` and PyTorch.
    - Hands-on examples with Fashion MNIST, a modern and more challenging alternative to the original MNIST dataset.

---

## Setup and Requirements

### 1. Prerequisites
Make sure you have the following installed:
- Python 3.7+
- Jupyter Notebook
- Required libraries (install via `pip`):
    ```bash
    pip install torch torchvision numpy matplotlib
    ```

### 2. Cloning the Repository
Clone the repository to your local machine:
```bash
git clone https://github.com/Mehrnaz-Sahebi/mytorch.git
```
### 3. Running the Notebooks
Launch Jupyter Notebook:
```bash
jupyter notebook
```
Then open any of the provided notebooks:
- `Conv_MNIST_mytorch.ipynb`
- `Conv_MNIST_pytorch.ipynb`
- `MLP_MNIST_pytorch.ipynb`
Follow the cells sequentially to execute the code.

## Results and Observations

- The `mytorch` library implements core functionality of PyTorch, such as:
  - Custom tensor operations
  - Automatic differentiation
  - Layer abstractions (e.g., linear, convolutional)

 ## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Authors  
- **Mehrnaz Sahebi** [GitHub Profile](https://github.com/Mehrnaz-Sahebi)
