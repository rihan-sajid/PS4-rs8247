import numpy as np
from scipy.sparse import diags, eye, kron

def solve_bvp_iterative(N, tol=1e-8, max_newton=50, max_jacobi=1000, jacobi_tol=1e-5):
    h = 1.0 / N
    M = N - 1
    u_int = np.ones(M * M)
    
    # 2D Laplacian Matrix
    main_diag = -2 * np.ones(M)
    off_diag = np.ones(M - 1)
    D1 = diags([off_diag, main_diag, off_diag], [-1, 0, 1])
    Laplacian_2D = (kron(eye(M), D1) + kron(D1, eye(M))) / h**2
    
    # Boundary vector
    bc = np.zeros((M, M))
    bc[0, :] += 1/h**2; bc[-1, :] += 1/h**2
    bc[:, 0] += 1/h**2; bc[:, -1] += 1/h**2
    bc = bc.flatten()
    
    # Initial guess for Jacobi step
    delta_u = np.zeros_like(u_int) 
    
    for k in range(max_newton):
        F = Laplacian_2D.dot(u_int) + bc - u_int**4
        newton_res = np.max(np.abs(F))
        if newton_res < tol:
            break
            
        # Jacobian Diagonal and Off-Diagonal
        J_diag = Laplacian_2D.diagonal() - 4 * u_int**3
        
        # Jacobi Iteration for J * delta_u = -F
        for j_iter in range(max_jacobi):
            # Compute matrix-vector product for off-diagonal terms
            J_delta = Laplacian_2D.dot(delta_u) - 4 * u_int**3 * delta_u
            off_diag_mult = J_delta - J_diag * delta_u
            
            delta_u_new = (-F - off_diag_mult) / J_diag
            
            # Check Jacobi convergence
            if np.max(np.abs(delta_u_new - delta_u)) < jacobi_tol:
                delta_u = delta_u_new
                break
            delta_u = delta_u_new
            
        u_int += delta_u
        
    U_full = np.ones((N + 1, N + 1))
    U_full[1:N, 1:N] = u_int.reshape((M, M))
    return U_full

def compute_convergence():
    print("Computing N=512 (Ground Truth)...")
    u_512 = solve_bvp_iterative(512)
    print("Computing N=256...")
    u_256 = solve_bvp_iterative(256)
    print("Computing N=128...")
    u_128 = solve_bvp_iterative(128)
    
    # Subsample 512 to match 128 and 256 grids
    u_512_at_128 = u_512[::4, ::4]
    u_512_at_256 = u_512[::2, ::2]
    
    # Calculate L2 errors (normalized by number of points)
    err_128 = np.sqrt(np.sum((u_128 - u_512_at_128)**2) / (129**2))
    err_256 = np.sqrt(np.sum((u_256 - u_512_at_256)**2) / (257**2))
    
    print(f"\nError at N=128: {err_128:.2e}")
    print(f"Error at N=256: {err_256:.2e}")
    order = np.log2(err_128 / err_256)
    print(f"Empirical Order of Accuracy: {order:.2f} (Expected: ~2.0)")

if __name__ == "__main__":
    compute_convergence()