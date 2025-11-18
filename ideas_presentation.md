General grand Plan
# Project: Regression and Double Descent
## A Non-Linear Programming Capstone Project

### Project Overview

This repository explores the theory and practice of regression, tracing a path from foundational statistical methods to modern phenomena in machine learning. Using **polynomial regression** as a clear and controllable case study, this project investigates the core challenges of model fitting, complexity, optimization, and generalization.

While polynomial regression is linear in its *parameters*, it is a fundamental tool for modeling non-linear relationships. This project is central to the concepts of **Non-Linear Programming** as it revolves around a core task: finding the optimal set of parameters ($\beta$) that minimize a loss function (Mean Squared Error), and understanding *why* a simple minimization on training data is not always the best solution.

The project is structured in five parts, implemented in a single notebook, each building on the last.

---

### Part 1: Machine Learning & Regression Concepts (Complete)

This section establishes the fundamental problem of supervised learning: fitting a model to noisy data and evaluating its ability to generalize.

* **Objective:** To visually and quantitatively demonstrate the concepts of **underfitting** and **overfitting**.
* **Methodology:**
    1.  A "true" non-linear function ($f(x) = 0.5x^2$) is defined.
    2.  Noisy synthetic data is generated from this function ($y = f(x) + \epsilon$).
    3.  The data is split into **training** and **test** sets.
    4.  Polynomial models of varying degrees (e.g., 1, 2, 10, 20) are fit to the training data.
* **Key Concepts:** Supervised Learning, Regression, Generalization, Train/Test Split, Mean Squared Error (MSE), Underfitting (high-bias), Overfitting (high-variance).

---

### Part 2: Mathematical Foundations & Linear Algebra (Complete)

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

### Part 3: Regularization (Working atm)

This section introduces a classic technique to combat the overfitting observed in Part 1 by adding a penalty to the loss function.

* **Objective:** To control model complexity and prevent overfitting using **Ridge Regression (L2 Regularization)**.
* **Methodology:**
    1.  Modify the loss function to include a penalty term: $\text{Loss} = \text{MSE} + \lambda ||\beta||_2^2$.
    2.  Implement the analytical solution for Ridge: $\beta = (X^T X + \lambda I)^{-1} X^T y$.
    3.  Observe how the fitted model and its test error change as the regularization strength hyperparameter ($\lambda$) is varied.
* **Key Concepts:** Regularization, L2 Penalty (Ridge), Hyperparameter Tuning, Constrained Optimization, Bias-Variance Tradeoff.

---

### Part 4: Optimization in Overparameterized Models (in the future, waiting for part 3 be completed)

This section transitions from the analytical solutions of classical regression to the iterative optimization algorithms that power modern, complex models.

* **Objective:** To explore the behavior of models in the **overparameterized regime** (where parameters > data points) and understand the **implicit bias** of optimization algorithms.
* **Methodology:**
    1.  Set up an overparameterized system (e.g., degree 10 polynomial on 10 data points).
    2.  Instead of an analytical solution, use **Gradient Descent (GD)** (specifically, the Adam optimizer) to find the parameters $\beta$.
    3.  Train the model multiple times from different random initializations.
* **Key Concepts:** Gradient Descent, Iterative Optimization, Automatic Differentiation (Backpropagation), Overparameterized Systems (Interpolation), Loss Landscapes, **Implicit Bias**.

---

### Part 5: The Bias-Variance Tradeoff & Double Descent (in the future, waiting for part 3 and part 4 be completed)


This capstone section synthesizes all previous concepts to explore the true, and often counter-intuitive, relationship between model complexity and generalization error.

* **Objective:** To reproduce the classical **Bias-Variance Tradeoff** and discover the modern **Double Descent** phenomenon.
* **Methodology:**
    1.  Conduct a large-scale experiment, fitting models of increasing complexity (polynomial degree 1 through 50+) on many random data samples.
    2.  Empirically calculate and plot the **Bias²**, **Variance**, and **Total Test Error** as a function of model complexity (degree).
* **Expected Outcome:**
    1.  **Classical Regime:** We will first observe the classic U-shaped test error curve, where error is minimized at an optimal intermediate complexity.
    2.  **Modern Regime:** As complexity passes the **interpolation threshold** (where parameters ≈ data points), we will observe the test error *decreasing again*, revealing the double descent curve.
* **Key Concepts:** Bias-Variance Decomposition, Bias-Variance Tradeoff (Classical), Interpolation Threshold, Double Descent (Modern Generalization).




PNL Presentation Ideas
 - minimize the loss function is the goal of pure optimization. But in machine learning, we must minimize the loss while also ensuring our model generalizes to new data
- read Double Descent Demystified: Identifying, Interpreting &
Ablating the Sources of a Deep Learning Puzzle
- check https://gitlab.com/harvard-machine-learning/double-descent

- suggestions to improve:
   

### The Current Flaw in Alignment

The project designed to create the *exact* pathological problem that your textbook's algorithms are designed to solve:
1.  **Part 1** establishes the problem of overfitting.
2.  **Part 2** (Vandermonde Matrix) creates a loss function $J(\beta) = \|y - X\beta\|_2^2$ whose Hessian, $H = 2X^T X$, is **famously ill-conditioned**.
3.  This ill-conditioned Hessian creates a loss surface that is a long, narrow, "squashed" parabolic valley.

And yet, **Part 4 of the project completely ignores them.** It jumps directly to `Adam`, which is a **stochastic** optimizer. It is an algorithm from an entirely different domain (large-scale deep learning) that is not covered in the textbook.

### The Proposed Change to Align the Project

I would **replace the current Part 4** with a new, more rigorous "Part 4: Classical Iterative (NLP) Solvers."

The objective of this new section would be: "To solve the ill-conditioned least-squares problem using the full-batch, deterministic optimization algorithms from *Ribeiro & Karas* and to compare their practical performance."

The methodology would be to implement the following algorithms from scratch to find the optimal $\beta$ vector for your high-degree polynomial:

1.  **Full-Batch Gradient Descent (Steepest Descent, Ch. 5.1):**
    * **Implementation:** You would use the full-batch gradient we derived, $\nabla J(\beta) = 2(X^T X \beta - X^T y)$.
    * **What it will demonstrate:** You will be able to *visually* plot the path of $\beta$. You will see the characteristic, slow, **zig-zagging convergence** as it bounces off the walls of the narrow valley created by the ill-conditioned Vandermonde matrix. This is the textbook failure mode, brought to life.

2.  **Newton's Method (Ch. 5.2):**
    * **Implementation:** You would use the analytical Hessian $H = 2X^T X$.
    * **What it will demonstrate:** Since the problem is purely quadratic, Newton's method will solve for $\beta$ in **one single step**. This is the perfect empirical demonstration of its quadratic convergence and its theoretical power. It finds the minimum of the valley instantly.

3.  **Quasi-Newton Method (L-BFGS, Ch. 5.4):**
    * **Implementation:** You would implement the L-BFGS algorithm.
    * **What it will demonstrate:** You will see it converge dramatically faster than Gradient Descent (superlinear convergence) but in more than one step. It will serve as the perfect "compromise" algorithm, demonstrating how to "learn" the curvature of the valley without the cost of computing the full Hessian.

### Why This Change is a Profound Improvement

1.  **Direct Course Alignment:** It makes the project a direct application of Chapter 5 of your textbook.
2.  **Deeper Theoretical Insight:** It provides a direct, empirical comparison of the convergence rates (linear vs. superlinear vs. quadratic) that you are studying abstractly.
3.  **A More Complete Narrative:** It would provide a *complete* picture of the available solutions to ill-conditioned problems:
    * **Part 2 (Analytical):** Fails due to numerical instability (inverting $X^T X$).
    * **Part 3 (Regularization):** An *analytical trick* (adding $\lambda I$) to fix the instability.
    * **New Part 4 (Iterative):** A *family of algorithmic solutions* (GD, L-BFGS, Newton) that also solve the problem, each with a different performance trade-off.
