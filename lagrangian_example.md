## Simple Lagrange multiplier example — solved step by step

**Problem.** Find the rectangle of maximum area with fixed perimeter $P>0$.

Let the rectangle side lengths be $x>0$ and $y>0$. The area is
$$
A(x,y)=xy,
$$
and the perimeter constraint is
$$
2(x+y)=P \quad\Longleftrightarrow\quad x+y=\frac{P}{2}.
$$

We solve
$$
\max_{x,y>0}\; A(x,y)=xy
\qquad\text{s.t.}\qquad x+y=\tfrac{P}{2}.
$$

---

### 1. Form the Lagrangian
Introduce a Lagrange multiplier $\lambda$ and form
$$
\mathcal{L}(x,y,\lambda)=xy + \lambda\!\left(\tfrac{P}{2}-x-y\right).
$$
(The constraint is written as $\tfrac{P}{2}-x-y=0$.)

---

### 2. Stationarity conditions (first-order)
Take partial derivatives and set them to zero. We use an aligned block for clarity:
$$
\begin{aligned}
\frac{\partial\mathcal{L}}{\partial x} &= y - \lambda = 0, \\
\frac{\partial\mathcal{L}}{\partial y} &= x - \lambda = 0, \\
\frac{\partial\mathcal{L}}{\partial \lambda} &= \tfrac{P}{2} - x - y = 0.
\end{aligned}
$$

From the first two equations:
$$
y=\lambda,\qquad x=\lambda \quad\Longrightarrow\quad x=y.
$$

Plug into the constraint:
$$
x+x=\tfrac{P}{2}\quad\Rightarrow\quad x=\tfrac{P}{4},\qquad y=\tfrac{P}{4}.
$$

So the candidate solution is the square with side $P/4$.

---

### 3. Verify it's a maximum
We can check by reducing to a single-variable problem (substitution) or by second-derivative test on the reduced function.

Substitute $y=\tfrac{P}{2}-x$ into $A(x,y)$:
$$
A(x)=x\Big(\tfrac{P}{2}-x\Big) = -x^2 + \tfrac{P}{2}x.
$$
This is a concave quadratic with second derivative $A''(x)=-2<0$, so its critical point at
$$
x=\frac{P/2}{2}=\frac{P}{4}
$$
is a global maximum on the interval $0<x<P/2$. Thus the stationary point found by the Lagrangian is indeed the maximizer.

---

### 4. Result and interpretation
- **Optimal rectangle:** a square with sides $x=y=P/4$.  
- **Maximum area:** $A_{\max} = \left(\tfrac{P}{4}\right)^2 = \tfrac{P^2}{16}$.

**Interpretation of $\lambda$:** at the optimum $\lambda = x = y = P/4$. In general, the Lagrange multiplier measures the sensitivity of the optimal objective value to a small change in the constraint (here, how much the maximal area would increase per unit increase in the allowed half-perimeter).

---

### 5. Numerical example
If $P=20$, optimal sides $x=y=5$, maximum area $A_{\max}=25$.