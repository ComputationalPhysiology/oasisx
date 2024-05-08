# Splitting schemes

Author: Jørgen S. Dokken

As described in {cite}`Oasis-2015` Oasisx uses a fractional step method for solving the Navier-Stokes equations.
This means that we are solving the set of equations:

Find $\mathbf{u}\in \mathbf{V}_h, p \in \mathbf{Q}$ such that over $\Omega\subset \mathbb{R}^d$

$$
\begin{align}
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u}\cdot \nabla)\mathbf{u} &= \nu \nabla^2 \mathbf{u} - \nabla p + \mathbf{f} &\text{ in } \Omega\\
\nabla \cdot \mathbf{u} &= 0& \text { in } \Omega \\
\quad\nu\frac{\partial \mathbf{u}}{\partial\mathbf{n}} - p\mathbf{n} &= h\mathbf{n}& \text{ on }\partial \Omega_N \\
\quad\mathbf{u}&=\mathbf{g}&\text{ on } \partial \Omega_D
\end{align}
$$

where $\mathbf{u} = (u_1(\mathbf{x}, t), \dots, u_d(\mathbf{x}, t))$ is the velocity vector, $\nu$ the kinematic viscosity, $p(\mathbf{x}, t)$ the fluid pressure and $\mathbf{f}(\mathbf{x}, t)$ are the volumetric forces. The fluid density is incorporated with the pressure $p$.
We use the a pseudo-traction boundary condition on $\partial\Omega_N$, where $h=0$ corresponds to the natural boundary condition. We use $\frac{\partial{\cdot}}{\partial \mathbf{n}}= \mathbf{n}^T(\nabla\cdot)$.

We assume that $\partial\Omega=\partial\Omega_N\cup \partial \Omega_D$, $\partial \Omega_N \cap \partial \Omega_D = \emptyset$. If $\partial \Omega_N = \emptyset$ we have the additional constraints

$$
\begin{align}
\int_\Omega p ~\mathrm{d}x &= 0\\
\int_{\partial\Omega}\mathbf{g}\cdot \mathbf{n}~\mathrm{d}s &= 0
\end{align}
$$

For the initial condition, we have that $\mathbf{u}(x, 0)=\mathbf{u}_0$, where $\nabla \cdot \mathbf{u}_0=0$ and $\mathbf{u}_0\cdot \mathbf{n} = \mathbf{g}(x,0)\cdot \mathbf{n}$.

## Stokes equation

The following section will follow a similar derivation as done by Timmermans {cite}`timmermans1996`.
We start by considering a simpler problem, namely solving

$$
\begin{align}
\frac{\partial \mathbf{u}}{\partial t} - \nu \Delta \mathbf{u} + \nabla p &= f &&\text{ in } \Omega\\
\nabla \cdot \mathbf{u} &= 0 &&\text{ in } \Omega\\
\mathbf{u} &= \mathbf{g}(x,t) &&\text{ on } \partial \Omega_D\\
\nu \frac{\partial \mathbf{u}}{\partial n} - p \mathbf{n} &= \mathbf{h} &&\text{ on }\partial\Omega_N
\end{align}
$$

We use a Crank-Nicolson discretization in time, and thus want to solve

$$
\begin{align}
\frac{\mathbf{u}^n-\mathbf{u}^{n-1}}{\Delta t} - \frac{\nu}{2}\Delta(\mathbf{u}^n+\mathbf{u}^{n-1}) +\nabla p^{n-\frac{1}{2}} &= f^{n-\frac{1}{2}} && \text{ in } \Omega \\
\nabla \cdot u^n &= 0 && \text{ in } \Omega \\
\mathbf{u}^n &=\mathbf{g}^n && \text{ on } \partial \Omega_D\\
\frac{\nu}{2}\frac{\partial (\mathbf{u}^n + \mathbf{u}^{n-1})}{\partial n} - p^{n-\frac{1}{2}}\mathbf{n} &= \mathbf{h}^{n-\frac{1}{2}}&& \text{ on } \partial \Omega_N
\end{align}
$$

However, we do not want to solve this coupled set of equations, and instead solve for $\mathbf{u}^n$ and $p^{n-\frac{1}{2}}$ in a segregated fashion.
We start by selecting a $p^\star$ such that $p^\star = p^{n-\frac{1}{2}} + \mathcal{O}(\Delta t)$.
A common choice is to use $p^\star= p^{n-\frac{3}{2}}$.

We next solve the following problem

$$
\begin{align}
\frac{\mathbf{u}^\star - \mathbf{u}^{n-1}}{\Delta t} - \frac{\nu}{2}\Delta \left(\mathbf{u}^* +\mathbf{u}^{n-1}\right) &= - \nabla p^\star + f^{n-\frac{1}{2}} && \text{ in } \Omega \\
\mathbf{u}^* &= \mathbf{g}^n && \text{ on } \partial \Omega_D \\
\frac{\nu}{2}\frac{\partial(\mathbf{u}^* + \mathbf{u}^{n-1})}{\partial n} - p^\star \mathbf{n} &= \mathbf{h}^{n-\frac{1}{2}} && \text{ on }\partial \Omega_N
\end{align}
$$

We subtract the equation for $\mathbf{u^*}$ from the equation for $\mathbf{u}^n$ to obtain

$$
\begin{align}
\frac{\mathbf{u}^n - \mathbf{u}^\star}{\Delta t} - \frac{\nu}{2}\Delta (\mathbf{u}^n - \mathbf{u}^\star) &= - \nabla (p^{n-\frac{1}{2}}- p^\star)&& \text{ in } \Omega
\end{align}
$$

By taking the divergence of this equation we obtain

$$
\begin{align}
\frac{1}{\Delta t} \nabla \cdot (\mathbf{u}^n - \mathbf{u}^\star)-\frac{\nu}{2} \nabla \cdot (\Delta u^n - \Delta u^{*})  &= - \nabla \cdot \nabla (p^{n-\frac{1}{2}}- p^\star)&& \text{ in } \Omega
\end{align}
$$

We use the fact that $\nabla \cdot \mathbf{u}^n = 0$ and the identitiy $\nabla \cdot \Delta \mathbf{T} = \nabla \cdot \nabla (\nabla \cdot \mathbf{T})- \nabla \cdot (\nabla \times(\nabla \times \mathbf{T}) ) =# \Delta (\nabla \cdot \mathbf{T})$ as $\nabla \cdot (\nabla \times L) = 0 \quad \forall L$ we can simplify our equation

$$
\begin{align}
&-\frac{1}{\Delta t}\nabla \cdot \mathbf{u}^\star- \frac{\nu}{2} \Delta (\nabla \cdot \mathbf{u}^n - \nabla \cdot \mathbf{u}^\star) - = -\Delta (p^{n-\frac{1}{2}}-p^\star)\\
&= -\frac{1}{\Delta t}\nabla \cdot \mathbf{u}^\star +\Delta \left(\frac{\nu}{2} \nabla \cdot \mathbf{u}^*\right)
\end{align}
$$

which means that we can conclude with

$$
\Delta \left(p^{n-\frac{1}{2}}-p^*+\frac{\nu}{2}\nabla \cdot \mathbf{u}^*\right) = \frac{1}{\Delta t}\nabla \cdot \mathbf{u}^*
$$

Setting $\phi = p^{n-\frac{1}{2}} - p^* + \frac{\nu}{2}\nabla \cdot \mathbf{u}^\star$.
We can solve this Poisson-type problem for $\phi$.
We can then project the pressure $p^{n-\frac{1}{2}}=p^*+\phi-\frac{\nu}{2}\nabla \cdot \mathbf{u}^*$.

To get an expression for $\mathbf{u}^n$ we use that $\mathbf{u}^n = \mathbf{u}^\star + D$ and that

$$
\begin{align}
\nabla \cdot \mathbf{u}^{n} &= 0\\
&= \nabla \cdot \mathbf{u}^\star + \nabla \cdot D
\end{align}
$$

From the pressure correction equation we have that $\nabla \cdot \mathbf{u}^\star = \Delta t \Delta \phi= \Delta t \nabla \cdot \nabla \phi$. Thus by setting $D=-\Delta t \nabla \phi$ we have that $\mathbf{u}^{n}$ is divergence free.

Concluding we solve the following equations

### Tentative velocity

$$
\begin{align}
\frac{\mathbf{u}^\star - \mathbf{u}^{n-1}}{\Delta t} - \frac{\nu}{2}\Delta \left(\mathbf{u}^* +\mathbf{u}^{n-1}\right) &= - \nabla p^\star + f^{n-\frac{1}{2}} && \text{ in } \Omega \\
\mathbf{u}^* &= \mathbf{g}^n && \text{ on } \partial \Omega_D \\
\frac{\nu}{2}\frac{\partial(\mathbf{u}^* + \mathbf{u}^{n-1})}{\partial n} - p^\star \mathbf{n} &= \mathbf{h}^{n-\frac{1}{2}} && \text{ on }\partial \Omega_N
\end{align}
$$

### Pressure correction

$$
\begin{align}
\Delta \phi &= \frac{1}{\Delta t}\nabla \cdot \mathbf{u}^\star& \text{in }\Omega\\
p^{n-\frac{1}{2}}&=p^*+\phi-\frac{\nu}{2}\nabla \cdot \mathbf{u}^*\\
\end{align}
$$

### Velocity correction

$$
\mathbf{u}^n = \mathbf{u}^*- \Delta t \nabla \phi
$$

### Essential boundary conditions

We note that we have not specified boundary conditions for the pressure correction.

Assume that $\partial \Omega_N=\emptyset$, we then use that $u^*=\mathbf{g}^n$ on the whole boundary, and the flux condition of $\mathbf{g}$ over $\partial\Omega$.
We integrate the pressure correction equation over $\Omega$ (using the divergence theorem)

$$
\begin{align}
\int_\Omega \nabla \cdot \nabla \phi~\mathrm{dx} &= \int_{\partial\Omega} \frac{\partial \phi}{\partial n}~\mathrm{d}s\\
&= \int_\Omega \frac{1}{\Delta t} \nabla \cdot \mathbf{u}^* = \frac{1}{\Delta t}\int_{\partial\Omega}\mathbf{u}^*\cdot \mathbf{n}~\mathrm{d}s =
\frac{1}{\Delta t}\int_{\partial\Omega}g^n\cdot \mathbf{n}~\mathrm{d}s = 0
\end{align}
$$

Thus we use that $\frac{\partial \phi}{\partial n}=0$ on $\partial \Omega$.

It has been discussed in many papers that one could use $\phi=0$ on $\partial\Omega_D$, see for instance chapter 10 of {cite}`guermond2006`. In {cite}`guermond2005` it is shown that by using the rotational form of the equations (i.e. including the divergence term in $\phi$) yield reasonable error estimates.

However, if $\partial \Omega_N\neq\emptyset$, then we need to consider the open boundary conditions for the pressure correction schemes, see the next section or {cite}`guermond2004rotational`

Also note that we have lost control of the tangential part of the corrected velocity, as we do not have that $\mathbf{u}^n\cdot \mathbf{t} = \mathbf{u}^\star \cdot \mathbf{t} - \Delta t \nabla \phi \cdot \mathbf{t}\neq\mathbf{g}^n$ as $\nabla \phi \cdot \mathbf{t}\neq 0$.

## Navier-Stokes equation

We split these coupled equations into a set of simpler equations by using a fractional step method, described in for instance {cite}`simo-1994`. We arrive at the following scheme

$$
\begin{align}
    \frac{u_k^{I}-  u_k^{n-1}}{\Delta t} + B_k^{n-\frac{1}{2}} &= \nu \nabla^2 \tilde u_k - \nabla_k p^\star + f_k^{n-\frac{1}{2}} & \text{for } k=1,\dots d&\text{ in } \Omega\\
    u_k^I &= g_k&& \text{ on } \partial \Omega_D\\
    \nu\frac{\partial \tilde u_k }{\partial  n} - p^*n_k &= h^{n-\frac{1}{2}}n_k&&\text{ on }\partial\Omega_N\\
    \nabla^2\phi &= -\frac{1}{\Delta t} \nabla \cdot \mathbf{u}^I, &&\text{ in } \Omega\\
    \frac{\partial\phi}{\partial n}&=0&& \text{ on }\partial \Omega_D \\
    \phi &= 0 &&\text{ on } \partial \Omega_N\\
    \frac{u_k^n-u_k^I}{\Delta t} &= -\frac{\partial}{\partial x_k}\phi & \text{for } k=1,\dots d,
\end{align}
$$

where $u_k^n$ is the $k$th component of the velocity vector at time $t^n$ $\phi = p^{n-\frac{1}{2}}-p^\star$ is a pressure correction, $p^\star$ the tentative pressure.

Using $\phi=0$ on $\partial\Omega_n$ can cause locking {cite}`poux2011` ,{cite}`guermond2006`, i.e. $p^{n+\frac{1}{2}}\vert_{\partial\Omega_N}=p^*\vert_{\partial\Omega_N}$.
We thus add an extra term

$$
\begin{align}
\phi_= p^{n-\frac{1}{2}}-p^\star - \xi \nu \nabla \cdot u_k^{I}
\end{align}
$$

where $0<\xi\leq 1$ as shown in {cite}`guermond2004rotational` and {cite}`timmermans1996`.

The first equation is solved for the tentative velocity $u_k^I$, where $\tilde u_k=\frac{1}{2}(u_k^I+u_k^{n-1})$, and the convective term $B_k^{n-\frac{1}{2}}=\mathbf{\bar{u}}\cdot \nabla \tilde u_k = (1.5 \mathbf{u}^{n-1}-0.5\mathbf{u}^{n-2})\cdot \nabla \tilde u_k$ is the implicit Adams-Bashforth discretization.

## Implementational aspects

We start by considering the tentative velocity step.

We use integration by parts and multiplication with a test function $v$ to obtain

$$
\begin{align}
    \frac{1}{\Delta t}\int_\Omega (u^I_k-u_k^{n-1}) v~\mathrm{d}x +& \int_\Omega \mathbf{\bar u} \cdot \frac{1}{2}\nabla (u_k^I + u_k^{n-1}) v ~\mathrm{d}x\\
    &+ \frac{\nu}{2}\int_\Omega \nabla (u_k^I + u_k^{n-1})\cdot \nabla v ~\mathrm{d}x \\
    &= -\int_\Omega p^\star \nabla_k v + f_k^{n-\frac{1}{2}}v ~\mathrm{dx} + \int_{\partial\Omega_N}h^{n-\frac{1}{2}}n_k \nabla_k v ~\mathrm{d}s.
\end{align}
$$

As $u_k^I$ is the unknown, we use $u_k^I=\sum_{i=0}^Mc_{k,i} \phi_i(\mathbf{x})$, where $c_{k,i}$ is the unknown coefficients, $\phi_i$ is the global basis functions of $u_k^I$.
We have that $u_k^{n-1}, u_k^{n-2}$ can be written as $u_k^{n-l}=\sum_{i=0}^M c_{k_i}i^{n-l} \phi_i$, where $c_i^{n-l}$ are the known coefficients from previous time steps.
We write $p^* = \sum_{q=0}^Qr_q\psi_q(\mathbf{x})$.
Summarizing, we have

$$
\left(\frac{1}{\Delta t} M + \frac{1}{2} C+ \frac{1}{2}\nu K\right) \mathbf{c}_k = \frac{1}{\Delta t} M \mathbf{c}_k^{n-1} -\frac{1}{2} C \mathbf{c}_k^{n-1} - \frac{1}{2}\nu K \mathbf{c}_k^{n-1} + P^k(p^*, f^{n-\frac{1}{2}}, h^{n-\frac{1}{2}})
$$

where

$$
\begin{align}
M_{ij} &= \int_\Omega \phi_j \phi_i ~\mathrm{d}x,\\
K_{ij} &= \int_\Omega \nabla \phi_j \cdot \nabla \phi_i ~\mathrm{d}x,\\
C_{ij} &= \int_\Omega \mathbf{\bar u}\cdot \nabla \phi_j \phi_i ~\mathrm{d}x,\\
P^k_j &=\int_\Omega \left(\sum_{q=0}^{Q}r\psi_q \right)\nabla_k\phi_j + f_k^{n-\frac{1}{2}}\phi_j~\mathrm{d}x
+\int_{\partial\Omega_N}h^{n-\frac{1}{2}}\phi_j~\mathrm{d}s
\end{align}.
$$

In Oasis {cite}`Oasis-2015`, one uses the fact that $M$, $K$ and $C$ is needed for the LHS of the variational problem, to avoid assembling them as vectors on the right hand side, and simply use matrix vector products and scaling to create the RHS vector from these pre-assembled matrices.

We also note that $M$ and $K$ are time independent, and thus only $C$ has to be assembled at every time step.

A difference between Oasis and the current implementation is that we implement the pressure condition as the natural boundary condition, and any supplied pressure $h$ will be part of the right hand side of the tentative velocity equation.

In Oasis, the choice of not including this term ment that one would have to re-assemble
$\int_\Omega \nabla_k p^*v~\mathrm{d}x$ in every inner iteration.
In the current implementation, we have that $\int_\Omega p^*\nabla_k v~\mathrm{d}x + \int_{\partial\Omega_N}h^{n-\frac{1}{2}}v~\mathrm{d}s$ has to be reassembled. As the boundary integral $\partial\Omega_N$ usually is small compared to the volume of the computational domain, this is a minimal increase in assembly time.

### Matrix-vector product

An algorithm for computing the matrix vector product follows

```python
M = assemble_mass_matrix()
K = assemble_stiffness_matrix()
for i in range(num_time_steps):
    # Compute convecting velocity for C
    for j in range(components):
        bar_u[j][:] = 0
        bar_u[j]+= 1.5 * u_1[j]
        bar_u[j]+= -0.5 * u_2[j]
    # Compute 0.5C
    A = assemble_C_matrix()
    A.scale(0.5)
    # Add mass matrix
    A.axpy(1/dt, M)

    #Add stiffness matrix
    A.axpy(-0.5*nu, K)

    # Compute matrix vector product and add to RHS
    for j in range(components):
        b[j] = body_forces
        b_tmp[j] = A * u_1[j]
        b[j]+= b_tmp[j]

    # Reset A for LHS
    A.scale(-1)
    A.xpy(2/dt, M)

    #
```

where $u_i$, $i=1,2$ is the solution at time-step $n-i$.

### Action method

As the matrix $A$ is sparse and potentially large (millions of degrees of freedom), we consider assembling `b` directly.

```python
M = assemble_mass_matrix()
K = assemble_stiffness_matrix()
for i in range(num_time_steps):
    # Compute convecting veloctiy for C
    for j in range(components):
        bar_u[j][:] = 0
        bar_u[j]+= 1.5 * u_1[j]
        bar_u[j]+= -0.5 * u_2[j]

    # Assemble RHS
    for j in range(components):
        b[j] = assemble_vector(body_forces[j] + mass_terms[j] \
            + stiffness_terms[j] + convective_terms[j])

    # Compute -0.5C
    A = assemble_C_matrix()
    A.scale(-0.5)
    # Add mass matrix
    A.axpy(-1/dt, M)

    #Add stiffness matrix
    A.axpy(0.5*nu, K)
```

In the next section, we will consider the performance differences for these two strategies

**References**

```{bibliography}
:filter: docname in docnames
```
