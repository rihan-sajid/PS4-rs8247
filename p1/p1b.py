import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import spsolve

def solve_bvp_direct(N=64, tol=1e-10, max_iter=50):
    # Grid spacing
    h = 1.0 / N
    
    # Number of interior points in one dimension
    M = N - 1
    
    # Initial guess for interior points (u=1 on boundary, let's guess u=1 everywhere)
    u_int = np.ones(M * M)
    
    # Standard 1D discrete Laplacian components
    main_diag = -2 * np.ones(M)
    off_diag = np.ones(M - 1)
    D1 = diags([off_diag, main_diag, off_diag], [-1, 0, 1])
    
    # 2D discrete Laplacian using Kronecker products
    I = eye(M)
    Laplacian_2D = (kron(I, D1) + kron(D1, I)) / h**2
    
    # Boundary condition contributions (u=1 on boundary)
    bc_vector = np.zeros((M, M))
    bc_vector[0, :] += 1/h**2  # Left
    bc_vector[-1, :] += 1/h**2 # Right
    bc_vector[:, 0] += 1/h**2  # Bottom
    bc_vector[:, -1] += 1/h**2 # Top
    bc_vector = bc_vector.flatten()
    
    # Newton-Raphson Loop
    for k in range(max_iter):
        # Construct F(U)
        # F(U) = Laplacian_2D * U + BC - U^4
        F = Laplacian_2D.dot(u_int) + bc_vector - u_int**4
        
        # Check convergence using L-infinity norm
        error = np.max(np.abs(F))
        print(f"Iteration {k}, Residual L-inf norm: {error:.2e}")
        if error < tol:
            print("Converged!")
            break
            
        # Construct Jacobian J(U)
        # J_ij = dF_i/dU_j = Laplacian_2D - 4*diag(U^3)
        J_nonlinear = diags(-4 * u_int**3)
        J = Laplacian_2D + J_nonlinear
        
        # Solve J * delta_U = -F
        delta_u = spsolve(J, -F)
        
        # Update U
        u_int += delta_u
        
    # Reconstruct the full 2D solution including boundaries
    U_full = np.ones((N + 1, N + 1))
    U_full[1:N, 1:N] = u_int.reshape((M, M))
    
    # Plotting
    X, Y = np.meshgrid(np.linspace(0, 1, N+1), np.linspace(0, 1, N+1))
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, U_full, levels=50, cmap='viridis')
    plt.colorbar(contour)
    plt.title(f'Elliptic BVP Solution (Direct Solver, N={N})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('problem_1b_solution.png')
    plt.show()

if __name__ == "__main__":
    solve_bvp_direct(N=64)