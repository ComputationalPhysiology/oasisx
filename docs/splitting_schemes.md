# Splitting schemes
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

A consequence of this is that $p^{n+\frac{1}{2}}\vert_{\partial\Omega_N}=p^*\vert_{\partial\Omega_N}$.

The first equation is solved for the tentative velocity $u_k^I$, where $\tilde u_k=\frac{1}{2}(u_k^I+u_k^{n-1})$, and the convective term $B_k^{n-\frac{1}{2}}=\mathbf{\bar{u}}\cdot \nabla \tilde u_k = (1.5 \mathbf{u}^{n-1}-0.5\mathbf{u}^{n-2})\cdot \nabla \tilde u_k$ is the implicit Adams-Bashforth discretization.

## Implementational aspects
We start by considering the tentative velocity step.

We use integration by parts and multiplication with a test function $v$ to obtain

$$
\begin{align}
    \frac{1}{\Delta t}\int_\Omega (u^I_k-u_k^{n-1}) v~\mathrm{d}x +& \int_\Omega \mathbf{\bar u} \cdot \frac{1}{2}\nabla (u_k^I + u_k^{n-1}) v ~\mathrm{d}x\\
    &+ \frac{\nu}{2}\int_\Omega \nabla (u_k^I + u_k^{n-1})\cdot \nabla v ~\mathrm{d}x \\
    &= \int_\Omega p^\star \nabla_k v + f_k^{n-\frac{1}{2}}v ~\mathrm{dx} + \int_{\partial\Omega_N}h^{n-\frac{1}{2}}n_kv~\mathrm{d}s.
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