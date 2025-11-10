# Project: Regression and Double Descent
## A Non-Linear Programming Capstone Project

### Project Overview

This repository explores the theory and practice of regression, tracing a path from foundational statistical methods to modern phenomena in machine learning. Using **polynomial regression** as a clear and controllable case study, this project investigates the core challenges of model fitting, complexity, optimization, and generalization.

While polynomial regression is linear in its *parameters*, it is a fundamental tool for modeling non-linear relationships. This project is central to the concepts of **Non-Linear Programming** as it revolves around a core task: finding the optimal set of parameters ($\beta$) that minimize a loss function (Mean Squared Error), and understanding *why* a simple minimization on training data is not always the best solution.

The project is structured in five parts, implemented in a single notebook, each building on the last.

---

### Part 1: Machine Learning & Regression Concepts

This section establishes the fundamental problem of supervised learning: fitting a model to noisy data and evaluating its ability to generalize.

* **Objective:** To visually and quantitatively demonstrate the concepts of **underfitting** and **overfitting**.
* **Methodology:**
    1.  A "true" non-linear function ($f(x) = 0.5x^2$) is defined.
    2.  Noisy synthetic data is generated from this function ($y = f(x) + \epsilon$).
    3.  The data is split into **training** and **test** sets.
    4.  Polynomial models of varying degrees (e.g., 1, 2, 10, 20) are fit to the training data.
* **Key Concepts:** Supervised Learning, Regression, Generalization, Train/Test Split, Mean Squared Error (MSE), Underfitting (high-bias), Overfitting (high-variance).

---

### Part 2: Mathematical Foundations & Linear Algebra

This section moves from the "what" to the "how," exploring the mathematical formulation of polynomial regression as a system of linear equations ($X\beta = y$).

* **Objective:** To understand the numerical methods used to solve for the model's coefficients ($\beta$) and the importance of the chosen basis.
* **Methodology:**
    1.  Construct the design matrix $X$ using a standard **power basis** (the **Vandermonde matrix**).
    2.  Explore alternative **basis functions**, such as the numerically stable **Legendre polynomials**.
    3.  Solve for $\beta$ using two different analytical methods:
        * The **Normal Equation**: $\beta = (X^T X)^{-1} X^T y$
        * The **Moore-Penrose Pseudoinverse**: $\beta = X^+ y$
* **Key Concepts:** Linear Basis Function Models, Vandermonde Matrix, Orthogonal Polynomials, Linear Least Squares, Numerical Stability, Ill-Conditioned Matrices, Pseudoinverse.

---

### Part 3: Regularization

This section introduces a classic technique to combat the overfitting observed in Part 1 by adding a penalty to the loss function.

* **Objective:** To control model complexity and prevent overfitting using **Ridge Regression (L2 Regularization)**.
* **Methodology:**
    1.  Modify the loss function to include a penalty term: $\text{Loss} = \text{MSE} + \lambda ||\beta||_2^2$.
    2.  Implement the analytical solution for Ridge: $\beta = (X^T X + \lambda I)^{-1} X^T y$.
    3.  Observe how the fitted model and its test error change as the regularization strength hyperparameter ($\lambda$) is varied.
* **Key Concepts:** Regularization, L2 Penalty (Ridge), Hyperparameter Tuning, Constrained Optimization, Bias-Variance Tradeoff.

---

### Part 4: Optimization in Overparameterized Models

This section transitions from the analytical solutions of classical regression to the iterative optimization algorithms that power modern, complex models.

* **Objective:** To explore the behavior of models in the **overparameterized regime** (where parameters > data points) and understand the **implicit bias** of optimization algorithms.
* **Methodology:**
    1.  Set up an overparameterized system (e.g., degree 10 polynomial on 10 data points).
    2.  Instead of an analytical solution, use **Gradient Descent (GD)** (specifically, the Adam optimizer) to find the parameters $\beta$.
    3.  Train the model multiple times from different random initializations.
* **Key Concepts:** Gradient Descent, Iterative Optimization, Automatic Differentiation (Backpropagation), Overparameterized Systems (Interpolation), Loss Landscapes, **Implicit Bias**.

---

### Part 5: The Bias-Variance Tradeoff & Double Descent

This capstone section synthesizes all previous concepts to explore the true, and often counter-intuitive, relationship between model complexity and generalization error.

* **Objective:** To reproduce the classical **Bias-Variance Tradeoff** and discover the modern **Double Descent** phenomenon.
* **Methodology:**
    1.  Conduct a large-scale experiment, fitting models of increasing complexity (polynomial degree 1 through 50+) on many random data samples.
    2.  Empirically calculate and plot the **Bias²**, **Variance**, and **Total Test Error** as a function of model complexity (degree).
* **Expected Outcome:**
    1.  **Classical Regime:** We will first observe the classic U-shaped test error curve, where error is minimized at an optimal intermediate complexity.
    2.  **Modern Regime:** As complexity passes the **interpolation threshold** (where parameters ≈ data points), we will observe the test error *decreasing again*, revealing the double descent curve.
* **Key Concepts:** Bias-Variance Decomposition, Bias-Variance Tradeoff (Classical), Interpolation Threshold, Double Descent (Modern Generalization).
