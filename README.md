# NeuraCore-C

A low-level implementation of artificial neural network (ANN) forward propagation built entirely from scratch in C99. This repository demonstrates memory allocation mechanics and linear algebra foundational layers behind modern deep learning.

## ⚡ Key Architectural Features
* **Flat Array Memory Mapping:** Multi-dimensional weights are flattened into contiguous 1D memory blocks to optimize CPU cache lines.
* **Modular Layer Layout:** Easily scale input features and hidden layer structures via simple macro definitions.

## 📦 Compilation & Execution
```bash
gcc main.c neuron.c -o neuracore -lm
./neuracore
