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
