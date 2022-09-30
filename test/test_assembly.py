from IPython import embed
import dolfinx
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import numpy as np


def assembly(mesh, P: int, run: int, mat_options: bool):
    V = dolfinx.fem.FunctionSpace(mesh, ("CG", int(P)))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Define variational forms
    mass = ufl.inner(u, v) * ufl.dx
    stiffness = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

    u_ab = [dolfinx.fem.Function(V, name=f"u_ab{i}") for i in range(mesh.geometry.dim)]
    convection = ufl.inner(ufl.dot(ufl.as_vector(u_ab), ufl.nabla_grad(u)), v) * ufl.dx
    [u_abi.interpolate(lambda x: x[0]) for u_abi in u_ab]

    # Compile forms
    mass_form = dolfinx.fem.form(mass)
    stiffness_form = dolfinx.fem.form(stiffness)
    convection_form = dolfinx.fem.form(convection)

    # Create time independent matrices
    opts = PETSc.Options()
    opts.prefixPush("M")
    petsc_options = {"-mat_view": "::ascii_info"}
    for k, v in petsc_options.items():
        opts[k] = v
    opts.prefixPop()
    opts.prefixPush("K")
    for k, v in petsc_options.items():
        opts[k] = v
    opts.prefixPop()

    M = dolfinx.fem.petsc.create_matrix(mass_form)
    M.setOptionsPrefix("M")
    M.setFromOptions()
    if mat_options:
        M.setOption(PETSc.Mat.Option.SYMMETRY_ETERNAL, True)
        M.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES, True)
    dolfinx.fem.petsc.assemble_matrix(M, mass_form)
    M.assemble()
    if mat_options:
        M.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATIONS, False)

    K = dolfinx.fem.petsc.create_matrix(stiffness_form)
    K.setOptionsPrefix("K")
    K.setFromOptions()

    if mat_options:
        K.setOption(PETSc.Mat.Option.SYMMETRY_ETERNAL, True)
        K.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES, True)

    dolfinx.fem.petsc.assemble_matrix(K, stiffness_form)
    K.assemble()
    if mat_options:
        K.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATIONS, False)

    # Solution from previous time step
    u_1 = dolfinx.fem.Function(V)
    u_1.interpolate(lambda x: np.sin(x[0])*np.cos(x[1]))
    dt = 0.5
    nu = 0.3

    A = dolfinx.fem.petsc.create_matrix(mass_form)
    # if mat_options:
    #A.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATIONS, False)
    #A.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES, True)

    b = dolfinx.fem.Function(V)
    A.zeroEntries()
    dolfinx.fem.petsc.assemble_matrix(A, convection_form)
    A.assemble()
    with dolfinx.common.Timer(f"~{P} {run} Mat-vec product") as _:
        A.scale(-0.5)
        A.axpy(1./dt, M)
        A.axpy(-0.5*nu, K)
        A.mult(u_1.vector, b.vector)
        b.x.scatter_forward()

    dt_inv = dolfinx.fem.Constant(mesh, 1./dt)
    dt_inv.name = "dt_inv"
    nu_c = dolfinx.fem.Constant(mesh, nu)
    nu_c.name = "nu"
    lhs = dt_inv * mass - 0.5 * nu_c * stiffness - 0.5*convection
    lhs = dolfinx.fem.form(ufl.action(lhs, u_1))
    b_d = dolfinx.fem.Function(V)

    with dolfinx.common.Timer(f"~{P} {run} Action") as _:
        dolfinx.fem.petsc.assemble_vector(b_d.vector, lhs)
        b_d.x.scatter_reverse(dolfinx.la.ScatterMode.add)
        b_d.x.scatter_forward()
    # dolfinx.common.list_timings(mesh.comm, [dolfinx.common.TimingType.wall])
    assert np.allclose(b.x.array, b_d.x.array)
    t_matvec = dolfinx.common.timing(f"~{P} {run} Mat-vec product")
    t_action = dolfinx.common.timing(f"~{P} {run} Action")
    t_matvec_all = mesh.comm.allgather(t_matvec[1])
    t_action_all = mesh.comm.allgather(t_action[1])

    if mesh.comm.rank == 0:
        print(f"Degree {P}")
        print(f"Num velocity dofs {V.dofmap.index_map_bs * V.dofmap.index_map.size_global}")
        print(
            f"Mat-vec\n Min: {np.min(t_matvec_all):.3e}, Max: {np.min(t_matvec_all):.3e} Avg: {np.sum(t_matvec_all)/len(t_matvec_all):.3e}")
        print(
            f"Action \n Min: {np.min(t_action_all):.3e}, Max: {np.min(t_action_all):.3e} Avg: {np.sum(t_action_all)/len(t_action_all):.3e}")
    return V.dofmap.index_map_bs * V.dofmap.index_map.size_global, t_matvec_all, t_action_all


if __name__ == "__main__":
    Nx, Ny, Nz = 2, 2, 2
    repeat = 3
    min_degree, max_degree = 1, 3  # 4
    mat_options = True

    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, Nx, Ny, Nz)
    Ps = np.arange(min_degree, max_degree, dtype=np.int32)
    num_dofs = np.zeros(len(Ps))
    t_matvec = np.zeros((len(Ps), mesh.comm.size), dtype=np.float64)
    t_action = np.zeros((len(Ps), mesh.comm.size), dtype=np.float64)
    results = {}
    j = 0
    for i, P in enumerate(Ps):
        for k in range(repeat):
            dof, matvec, action = assembly(mesh, P, j, mat_options=mat_options)
            for t in matvec:
                results[j] = {"P": P, "num_dofs": dof, "method":
                              "matvec", "time": t, "procs": MPI.COMM_WORLD.size}
                j += 1
            for t in action:
                results[j] = {"P": P, "num_dofs": dof, "method":
                              "action", "time": t, "procs": MPI.COMM_WORLD.size}
                j += 1

    if MPI.COMM_WORLD.rank == 0:
        import pandas
        df = pandas.DataFrame.from_dict(results, orient="index")
        # df.to_csv(f"results.csv", mode="a")
        # df = pandas.read_csv(f"results.csv")
        import seaborn
        df["label"] = "P" + df["P"].astype(str) + " " + \
            df["num_dofs"].astype(str) + " \n Comms: " + df["procs"].astype(str)
        plot = seaborn.catplot(data=df, kind="swarm",  x="label", y="time", hue="method")
        plot.set(yscale="log")
        import matplotlib.pyplot as plt
        plt.grid()
        plt.savefig(f"test_{mat_options}.png")
