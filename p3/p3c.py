import numpy as np
import matplotlib.pyplot as plt

def initial_condition(x, y):
    return np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / (0.15)**2)

def advection_solvers(N, t_end=10.0):
    a, b = 1.0, 2.0
    dx = 1.0 / N
    dy = 1.0 / N
    
    # Enforce CFL < 1 for stability
    dt = 0.4 * min(dx/a, dy/b) 
    num_steps = int(np.ceil(t_end / dt))
    dt = t_end / num_steps # Adjust dt to exactly hit t_end
    
    mu = a * dt / dx
    nu = b * dt / dy
    
    x = np.linspace(0, 1, N, endpoint=False)
    y = np.linspace(0, 1, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Initialize grids
    u_ctu = initial_condition(X, Y)
    u_lw = initial_condition(X, Y)
    
    err_ctu_history = []
    err_lw_history = []
    time_history = []
    
    # Exact solution at any time modulo 1.0 (since integer advection speeds and domain length 1)
    def exact_sol(t):
        X_shift = (X - a * t) % 1.0
        Y_shift = (Y - b * t) % 1.0
        return initial_condition(X_shift, Y_shift)

    for n in range(num_steps):
        t = n * dt
        
        # --- CTU Scheme ---
        u_ctu_im1_j = np.roll(u_ctu, shift=1, axis=0)
        u_ctu_i_jm1 = np.roll(u_ctu, shift=1, axis=1)
        u_ctu_im1_jm1 = np.roll(u_ctu_im1_j, shift=1, axis=1)
        
        u_ctu = ((1 - mu) * (1 - nu) * u_ctu +
                 mu * (1 - nu) * u_ctu_im1_j +
                 (1 - mu) * nu * u_ctu_i_jm1 +
                 mu * nu * u_ctu_im1_jm1)
                 
        # --- Lax-Wendroff Scheme ---
        u_ip1 = np.roll(u_lw, shift=-1, axis=0)
        u_im1 = np.roll(u_lw, shift=1, axis=0)
        u_jp1 = np.roll(u_lw, shift=-1, axis=1)
        u_jm1 = np.roll(u_lw, shift=1, axis=1)
        u_ip1_jp1 = np.roll(u_ip1, shift=-1, axis=1)
        u_ip1_jm1 = np.roll(u_ip1, shift=1, axis=1)
        u_im1_jp1 = np.roll(u_im1, shift=-1, axis=1)
        u_im1_jm1 = np.roll(u_im1, shift=1, axis=1)
        
        u_lw = u_lw \
            - 0.5 * mu * (u_ip1 - u_im1) - 0.5 * nu * (u_jp1 - u_jm1) \
            + 0.5 * mu**2 * (u_ip1 - 2*u_lw + u_im1) \
            + 0.5 * nu**2 * (u_jp1 - 2*u_lw + u_jm1) \
            + 0.25 * mu * nu * (u_ip1_jp1 - u_ip1_jm1 - u_im1_jp1 + u_im1_jm1)
            
        # Logging errors
        if n % 10 == 0 or n == num_steps - 1:
            u_ex = exact_sol(t + dt)
            err_ctu = np.sqrt(np.sum((u_ctu - u_ex)**2) / N**2)
            err_lw = np.sqrt(np.sum((u_lw - u_ex)**2) / N**2)
            err_ctu_history.append(err_ctu)
            err_lw_history.append(err_lw)
            time_history.append(t + dt)
            
        # Plot contours at t=1.0 and t=10.0
        if np.isclose(t + dt, 1.0) or np.isclose(t + dt, 10.0):
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            axs[0].contourf(X, Y, u_ctu, levels=20)
            axs[0].set_title(f"CTU at t={t+dt:.1f} (N={N})")
            axs[1].contourf(X, Y, u_lw, levels=20)
            axs[1].set_title(f"Lax-Wendroff at t={t+dt:.1f} (N={N})")
            plt.savefig(f'contours_N{N}_t{int(t+dt)}.png')
            plt.close()
            
    return time_history, err_ctu_history, err_lw_history

if __name__ == "__main__":
    plt.figure(figsize=(10, 6))
    for N in [128, 256, 512]:
        print(f"Running simulation for N={N}...")
        t_hist, err_ctu, err_lw = advection_solvers(N)
        plt.plot(t_hist, err_ctu, label=f'CTU N={N}', linestyle='--')
        plt.plot(t_hist, err_lw, label=f'Lax-Wendroff N={N}')
        
    plt.yscale('log')
    plt.xlabel('Time (t)')
    plt.ylabel('L2 Error')
    plt.title('Error Evolution over Time for Advection Schemes')
    plt.legend()
    plt.savefig('advection_errors.png')
    plt.show()