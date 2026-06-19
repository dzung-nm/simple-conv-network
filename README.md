# Simple Convolutional Neural Network (CNN)

A lightweight, educational Convolutional Neural Network implementation in Rust for the MNIST dataset with full backpropagation support.

## Overview

This project demonstrates a complete implementation of a CNN (Convolutional Neural Network) in Rust, featuring:
- **Convolutional layers** with im2col optimization
- **Pooling layers** (Max and Average pooling)
- **Fully connected layers** with dropout regularization
- **Multiple activation functions** (ReLU, Sigmoid, Softmax)
- **Optimizations** (L1/L2 regularization, cache between forward and backward passes)
- **Support early stopping**
- **BLAS acceleration** for matrix operations
- **Parallelization** using Rayon for multi-threaded training

## Usage

See an example in main.rs for how to create and train a simple CNN on the MNIST dataset. 
The code includes loading the dataset, defining the network architecture, and training the model.

You can run it using the following command:

```bash
cargo run --release --bin simple-conv-network
```

There is a LeNet-5 implementation in the bin folder, which can be run using:

```bash
cargo run --release --bin mnist_lenet5
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.