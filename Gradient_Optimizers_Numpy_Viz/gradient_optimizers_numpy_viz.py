# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for 3D plotting
from collections import defaultdict
import time
import warnings
import os # Added for creating directory

# Suppress potential overflow warnings
# warnings.filterwarnings('ignore', message='overflow encountered in power')
# warnings.filterwarnings('ignore', message='invalid value encountered in true_divide')
# warnings.filterwarnings('ignore', message='invalid value encountered in sqrt')
# warnings.filterwarnings('ignore', message='divide by zero encountered in true_divide')

# --- Function Definitions (NumPy) ---

def rosenbrock_np(x):
    """Rosenbrock function (2D). Expects a 1D array/list [x1, x2]."""
    x = np.asarray(x, dtype=np.float64)
    # Ensure input is a 1D array with exactly 2 elements
    if x.ndim != 1 or x.shape[0] != 2:
        raise ValueError(f"Rosenbrock function defined here expects a 1D array of shape (2,), got shape {x.shape} and ndim {x.ndim}")
    x0 = x[0]
    x1 = x[1]
    # Add checks for NaN/Inf input
    if not np.all(np.isfinite(x)):
        # print(f"Warning: NaN/Inf input to rosenbrock_np: {x}") # Optional debug print
        return np.inf # Return Inf if input is invalid
    val = (1 - x0)**2 + 100 * (x1 - x0**2)**2
    # Return Inf if calculation results in NaN/Inf
    return val if np.isfinite(val) else np.inf


def grad_rosenbrock_np(x):
    """Gradient of the Rosenbrock function (2D)."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] != 2:
        raise ValueError("Gradient for Rosenbrock function defined here is 2D")
    # Add checks for NaN/Inf input
    if not np.all(np.isfinite(x)):
        # print(f"Warning: NaN/Inf input to grad_rosenbrock_np: {x}") # Optional debug print
        return np.array([np.nan, np.nan], dtype=np.float64) # Return NaN gradient if input is invalid
    dx1 = -2.0 * (1 - x[0]) - 400.0 * x[0] * (x[1] - x[0]**2)
    dx2 = 200.0 * (x[1] - x[0]**2)
    grad = np.array([dx1, dx2])
    # Return NaN if calculation results in NaN/Inf
    grad[~np.isfinite(grad)] = np.nan
    return grad

def quadratic_np(x):
    """Simple quadratic function (3D)."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] != 3:
        raise ValueError("Quadratic function defined here is 3D")
    target = np.array([1, 2, 3], dtype=np.float64)
    # Add checks for NaN/Inf input
    if not np.all(np.isfinite(x)):
        # print(f"Warning: NaN/Inf input to quadratic_np: {x}") # Optional debug print
        return np.inf
    val = np.sum((x - target)**2)
    return val if np.isfinite(val) else np.inf

def grad_quadratic_np(x):
    """Gradient of the simple quadratic function (3D)."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] != 3:
        raise ValueError("Gradient for Quadratic function defined here is 3D")
    target = np.array([1, 2, 3], dtype=np.float64)
     # Add checks for NaN/Inf input
    if not np.all(np.isfinite(x)):
        # print(f"Warning: NaN/Inf input to grad_quadratic_np: {x}") # Optional debug print
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    grad = 2.0 * (x - target)
    grad[~np.isfinite(grad)] = np.nan
    return grad

# --- Data Generation ---

def generate_initial_point(dim, low=-4, high=4):
    """Generates a random initial point."""
    return np.random.uniform(low, high, size=dim).astype(np.float64)

# --- Optimizer Implementation (NumPy) ---

def run_numpy_optimizers(func, grad_func, initial_point, dim, optimizers_to_run,
                         base_lr=0.001, iterations=1000, gradient_noise_std=0.0,
                         specific_lrs=None, print_freq=100):
    """
    Runs manually implemented optimizers using NumPy. Enhanced NaN/Inf handling.
    """
    results = defaultdict(lambda: {'path': [], 'time': 0.0, 'losses': []})
    epsilon = 1e-8 # For numerical stability

    if specific_lrs is None:
        specific_lrs = {}

    for name in optimizers_to_run:
        print(f"\n--- Running {name} ---")
        lr = specific_lrs.get(name, base_lr)
        if name == "Adadelta (NumPy)":
             print(f"Optimizer: {name}, Iterations: {iterations} (lr usually ignored)")
        else:
             print(f"Optimizer: {name}, Iterations: {iterations}, Learning Rate: {lr:.6f}")

        # --- Initialize parameters and optimizer state ---
        params = initial_point.copy()
        path = [params.copy()] # Store initial point

        # Calculate and store initial loss, checking for validity
        initial_loss = func(params)
        losses = [initial_loss]
        if not np.isfinite(initial_loss):
             print(f"Warning: Initial loss is {initial_loss} for {name}. Stopping.")
             results[name]['path'] = np.array(path)
             results[name]['losses'] = np.array(losses)
             results[name]['time'] = 0.0
             continue # Skip this optimizer if initial state is bad

        start_time = time.time()

        # Optimizer state initialization
        velocity = np.zeros_like(params, dtype=np.float64)
        beta_momentum = 0.9
        grad_squared_sum = np.zeros_like(params, dtype=np.float64)
        Eg = np.zeros_like(params, dtype=np.float64)
        Edx = np.zeros_like(params, dtype=np.float64)
        rho_adadelta = 0.95
        beta2_rmsprop = 0.99
        m = np.zeros_like(params, dtype=np.float64)
        v = np.zeros_like(params, dtype=np.float64)
        u = np.zeros_like(params, dtype=np.float64)
        beta1 = 0.9
        beta2 = 0.999
        t = 0 # Timestep

        # Print initial state
        if (0 + 1) % print_freq == 0 or 0 == 0:
            loss_str = f"{initial_loss:.8f}"
            params_str = np.round(params, 4)
            print(f"  Iter {0+1:>5}/{iterations+1}, Loss: {loss_str}, Params: {params_str}")


        # --- Optimization Loop ---
        for i in range(iterations):
            t += 1 # Increment timestep

            # Calculate gradient, check for NaN/Inf
            grad = grad_func(params)
            if not np.all(np.isfinite(grad)):
                print(f"Warning: NaN/Inf gradient encountered at iter {i+1}. Stopping {name}.")
                break # Stop if gradient calculation fails

            # Add optional gradient noise
            if gradient_noise_std > 0:
                noise = np.random.normal(0, gradient_noise_std, size=params.shape)
                grad += noise
                if not np.all(np.isfinite(grad)): # Check again after adding noise
                    print(f"Warning: NaN/Inf gradient after adding noise at iter {i+1}. Stopping {name}.")
                    break

            # --- Store previous parameters for potential rollback ---
            params_prev = params.copy()

            # --- Apply Optimizer Update Rule ---
            try:
                if name in ["BGD (NumPy)", "SGD without momentum (NumPy)", "Mini-batch GD (NumPy)"]:
                    update = lr * grad
                    params = params - update
                elif name == "SGD with momentum (NumPy)":
                    velocity = beta_momentum * velocity + lr * grad
                    update = velocity
                    params = params - update
                elif name == "NAG (NumPy)":
                    params_lookahead = params - beta_momentum * velocity
                    grad_lookahead = grad_func(params_lookahead)
                    if not np.all(np.isfinite(grad_lookahead)): raise ValueError("NaN/Inf in NAG lookahead gradient")
                    if gradient_noise_std > 0: grad_lookahead += np.random.normal(0, gradient_noise_std, size=params.shape)
                    if not np.all(np.isfinite(grad_lookahead)): raise ValueError("NaN/Inf in NAG lookahead gradient after noise")
                    velocity = beta_momentum * velocity + lr * grad_lookahead
                    update = velocity
                    params = params - update
                elif name == "Adagrad (NumPy)":
                    grad_squared_sum += grad * grad
                    adjusted_lr = lr / (np.sqrt(grad_squared_sum) + epsilon)
                    update = adjusted_lr * grad
                    params = params - update
                elif name == "Adadelta (NumPy)":
                     Eg = rho_adadelta * Eg + (1 - rho_adadelta) * grad * grad
                     delta_x_num = np.sqrt(Edx + epsilon)
                     delta_x_den = np.sqrt(Eg + epsilon)
                     safe_den = np.where(delta_x_den == 0, epsilon, delta_x_den)
                     delta_x = - (delta_x_num / safe_den) * grad
                     Edx = rho_adadelta * Edx + (1 - rho_adadelta) * delta_x * delta_x
                     update = delta_x
                     params = params + delta_x # Adadelta adds the update
                elif name == "RMSprop (NumPy)":
                     Eg = beta2_rmsprop * Eg + (1 - beta2_rmsprop) * grad * grad
                     sqrt_Eg = np.sqrt(np.maximum(Eg, 0)) + epsilon # Ensure non-negative sqrt arg
                     adjusted_lr = lr / sqrt_Eg
                     update = adjusted_lr * grad
                     params = params - update
                elif name == "Adam (NumPy)":
                     m = beta1 * m + (1 - beta1) * grad
                     v = beta2 * v + (1 - beta2) * (grad * grad)
                     # Bias correction must handle t=1 edge case for denominator
                     m_hat = m / (1 - beta1**t) if (1 - beta1**t) != 0 else m / epsilon
                     v_hat = v / (1 - beta2**t) if (1 - beta2**t) != 0 else v / epsilon
                     sqrt_v_hat = np.sqrt(np.maximum(v_hat, 0)) + epsilon # Ensure non-negative sqrt arg
                     update = lr * m_hat / sqrt_v_hat
                     params = params - update
                elif name == "AdaMax (NumPy)":
                     m = beta1 * m + (1 - beta1) * grad
                     u = np.maximum(beta2 * u, np.abs(grad))
                     m_hat = m / (1 - beta1**t) if (1 - beta1**t) != 0 else m / epsilon
                     update = (lr / (u + epsilon)) * m_hat
                     params = params - update
                elif name == "Nadam (NumPy)":
                     m = beta1 * m + (1 - beta1) * grad
                     v = beta2 * v + (1 - beta2) * (grad * grad)
                     # Bias correction
                     m_hat = m / (1 - beta1**t) if (1 - beta1**t) != 0 else m / epsilon
                     v_hat = v / (1 - beta2**t) if (1 - beta2**t) != 0 else v / epsilon
                     sqrt_v_hat = np.sqrt(np.maximum(v_hat, 0)) + epsilon
                     # Nadam moment update part
                     nadam_m_update = (1 - beta1) * grad / (1 - beta1**t) if (1 - beta1**t) != 0 else (1 - beta1) * grad / epsilon
                     nadam_m = beta1 * m_hat + nadam_m_update

                     update = lr * nadam_m / sqrt_v_hat
                     params = params - update
                else:
                     print(f"Warning: Optimizer '{name}' is not implemented.")
                     break # Exit optimizer loop

                # --- Check for NaN/Inf in parameters AFTER update ---
                if not np.all(np.isfinite(params)):
                    print(f"Warning: NaN/Inf parameters generated at iter {i+1}. Reverting and stopping {name}.")
                    params = params_prev # Revert to previous valid state
                    break # Stop this optimizer

                # --- Calculate and check loss for the new valid state ---
                current_loss = func(params)
                if not np.isfinite(current_loss):
                     print(f"Warning: NaN/Inf loss generated at iter {i+1}. Reverting and stopping {name}.")
                     params = params_prev # Revert to previous valid state
                     break # Stop this optimizer

                # --- Store valid state ---
                path.append(params.copy())
                losses.append(current_loss)

                # --- Print status periodically ---
                if (i + 1) % print_freq == 0 or i == iterations - 1:
                   loss_str = f"{current_loss:.8f}"
                   params_str = np.round(params, 4)
                   print(f"  Iter {i+1:>5}/{iterations}, Loss: {loss_str}, Params: {params_str}")

            except Exception as e:
                print(f"Error during optimizer update for {name} at iteration {i+1}: {e}")
                import traceback
                traceback.print_exc()
                params = params_prev # Revert on any other exception during update
                break # Stop this optimizer

        # --- End Optimization Loop ---
        end_time = time.time()
        final_loss = func(params) # Loss at the final valid parameters
        results[name]['path'] = np.array(path)
        # Ensure losses array matches the final valid path length
        results[name]['losses'] = np.array(losses[:len(path)])
        results[name]['time'] = end_time - start_time

        print(f"--- {name} Finished ---")
        print(f"Total Iterations Run: {len(path) - 1}") # Report actual iterations completed
        print(f"Time: {results[name]['time']:.4f}s")
        loss_str = f"{final_loss:.8f}" if np.isfinite(final_loss) else "NaN/Inf"
        params_str = np.round(params, 5) if np.all(np.isfinite(params)) else "[NaN/Inf]"
        print(f"Final Loss: {loss_str}")
        print(f"Final Params: {params_str}")
        if not np.all(np.isfinite(params)):
             print("Warning: Final parameters may contain NaN/Inf values.")


    return results


# --- Visualization Functions ---

# Create directory for saving plots if it doesn't exist
output_dir = "optimization_plots"
os.makedirs(output_dir, exist_ok=True)

def plot_rosenbrock_paths(results_np, initial_point, target_minimum=(1,1), filename="rosenbrock_paths.png"):
    """
    Visualizes optimization paths on the Rosenbrock function (2D).
    Uses plt.subplots_adjust and fig.legend for explicit legend placement
    without resizing the axes plot area.
    """
    fig, ax = plt.subplots(figsize=(10, 8)) # Figure size might need adjustment

    # --- Contour plot setup (same as before) ---
    x_min_plot = -2.5
    x_max_plot = 2.5
    y_min_plot = -1.5
    y_max_plot = 4.0
    all_paths_flat = np.concatenate([data['path'] for data in results_np.values() if len(data['path']) > 0])
    if len(all_paths_flat) > 0:
        valid_paths = all_paths_flat[np.all(np.isfinite(all_paths_flat), axis=1)]
        if len(valid_paths)>0:
             x_min_plot = min(x_min_plot, valid_paths[:,0].min() - 0.5)
             x_max_plot = max(x_max_plot, valid_paths[:,0].max() + 0.5)
             y_min_plot = min(y_min_plot, valid_paths[:,1].min() - 0.5)
             y_max_plot = max(y_max_plot, valid_paths[:,1].max() + 1.0)

    x_range = np.linspace(x_min_plot, x_max_plot, 300)
    y_range = np.linspace(y_min_plot, y_max_plot, 300)
    X1, X2 = np.meshgrid(x_range, y_range)
    Z = (1 - X1)**2 + 100 * (X2 - X1**2)**2
    Z_safe = np.maximum(Z, 1e-12)
    levels = np.logspace(np.log10(max(Z_safe.min(), 1e-8)), np.log10(Z_safe.max()), 30)
    contour = ax.contourf(X1, X2, Z, levels=levels, cmap='viridis_r', alpha=0.7, norm=plt.matplotlib.colors.LogNorm())
    fig.colorbar(contour, ax=ax, label='log(f(x1, x2))') # Keep colorbar associated with axes
    ax.contour(X1, X2, Z, levels=levels, colors='gray', linewidths=0.5, alpha=0.6, norm=plt.matplotlib.colors.LogNorm())

    # --- Plot paths (same as before) ---
    markers = ['o', 's', 'p', '*', '+', 'x', 'D', 'v', '^', '<', '>']
    idx = 0
    for name, data in results_np.items():
        if not len(data['path']): continue
        path = np.array(data['path'])
        valid_path = path[np.all(np.isfinite(path), axis=1)]
        if len(valid_path) == 0: continue
        markevery=max(1, len(valid_path)//15)
        # Plot lines and markers on the axes object 'ax'
        ax.plot(valid_path[:, 0], valid_path[:, 1], marker=markers[idx % len(markers)], markevery=markevery, markersize=4, linestyle='--', label=f"{name} ({data['time']:.2f}s)", alpha=0.8, linewidth=1.5)
        ax.plot(valid_path[0, 0], valid_path[0, 1], 'go', markersize=6, label='_nolegend_')
        if len(valid_path) > 1:
             ax.plot(valid_path[-1, 0], valid_path[-1, 1], 'ro', markersize=6, label='_nolegend_')
        idx += 1

    # --- Plot target and set labels/title on 'ax' ---
    ax.plot(target_minimum[0], target_minimum[1], 'kx', markersize=12, mew=3, label=f'Minimum {target_minimum}') # Label needed for fig.legend
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    start_point_str = f"({initial_point[0]:.2f}, {initial_point[1]:.2f})" if np.all(np.isfinite(initial_point)) else "[NaN/Inf]"
    ax.set_title(f"Optimization Paths on Rosenbrock Function\nStart: {start_point_str}")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlim(x_range[0], x_range[-1])
    ax.set_ylim(y_range[0], y_range[-1])
    # Optional: Make aspect ratio equal if desired and range allows
    # ax.set_aspect('equal', adjustable='box')


    # --- Explicit Legend Placement using fig.legend ---
    # 1. Get handles and labels from the axes
    handles, labels = ax.get_legend_handles_labels()

    # 2. Adjust subplot parameters to make space on the right
    #    Leaves 25% of the figure width on the right for the legend
    plt.subplots_adjust(left=0.1, right=0.75, top=0.9, bottom=0.1)

    # 3. Place the legend on the figure in the reserved space
    if handles: # Check if there's anything to put in the legend
        fig.legend(handles, labels,           # Items to include
                   loc='upper left',          # Anchor point on the legend box
                   bbox_to_anchor=(0.77, 0.9), # Coordinates (figure units) to place the anchor point
                                              # (0.77 starts it in the reserved space, 0.9 near the top)
                   borderaxespad=0.,          # Padding
                   fontsize='medium'          # Font size
                   )

    # --- Save the figure ---
    full_filename = os.path.join(output_dir, filename)
    print(f"Saving plot to {full_filename}...")
    try:
        # Save the figure - bbox_inches='tight' might slightly counteract subplots_adjust,
        # but usually works well to remove excess whitespace around the whole figure.
        fig.savefig(full_filename, dpi=300, bbox_inches='tight')
        # If 'tight' causes issues, try saving without it:
        # fig.savefig(full_filename, dpi=300)
    except Exception as e:
        print(f"Error saving plot {full_filename}: {e}")

    plt.show()

def plot_rosenbrock_convergence(results_np, target_minimum=(1, 1), filename="rosenbrock_convergence.png"):
    """Visualizes convergence for the 2D Rosenbrock function (Loss and Distance)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    target_minimum = np.array(target_minimum, dtype=np.float64)

    # Plot 1: Function Value (Loss) vs Iteration
    ax1 = axes[0]
    min_loss_overall = float('inf')
    max_iter_plotted = 0
    for name, data in results_np.items():
         if not len(data['losses']): continue
         losses = np.array(data['losses'])
         valid_indices = np.isfinite(losses)
         valid_losses = losses[valid_indices]
         valid_iterations = np.arange(len(losses))[valid_indices]

         if len(valid_losses) == 0: continue # Skip if no valid losses
         max_iter_plotted = max(max_iter_plotted, len(losses))
         # Add epsilon for log scale if loss reaches zero or below
         plot_losses = np.maximum(valid_losses, 1e-12)
         min_loss_overall = min(min_loss_overall, plot_losses.min())
         ax1.plot(valid_iterations, plot_losses, label=f"{name}", alpha=0.8)

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Function Value (Loss) f(x)")
    ax1.set_title("Rosenbrock Convergence: Function Value")
    ax1.set_yscale('log')
    if np.isfinite(min_loss_overall): # Set ylim only if valid losses exist
       ax1.set_ylim(bottom=max(min_loss_overall / 10, 1e-10))
    if max_iter_plotted > 0:
        ax1.set_xlim(left=-0.05*max_iter_plotted, right=max_iter_plotted*1.05) # Adjust xlim based on iterations
    ax1.grid(True, which="both", ls=":", alpha=0.6)


    # Plot 2: Distance to Minimum vs Iteration
    ax2 = axes[1]
    min_dist_overall = float('inf')
    max_iter_plotted_dist = 0
    for name, data in results_np.items():
         if not len(data['path']): continue
         path = np.array(data['path'])
         valid_indices = np.all(np.isfinite(path), axis=1)
         valid_path = path[valid_indices]
         valid_iterations = np.arange(len(path))[valid_indices]

         if len(valid_path) == 0: continue
         max_iter_plotted_dist = max(max_iter_plotted_dist, len(path))

         distances = np.linalg.norm(valid_path - target_minimum, axis=1)
         # Add epsilon for log scale if distance reaches zero
         plot_distances = np.maximum(distances, 1e-12)
         min_dist_overall = min(min_dist_overall, plot_distances.min())
         ax2.plot(valid_iterations, plot_distances, label=f"{name} ({data['time']:.2f}s)", alpha=0.8)

    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Distance to Minimum ||x - x*||")
    ax2.set_title("Rosenbrock Convergence: Distance to Minimum")
    ax2.set_yscale('log')
    if np.isfinite(min_dist_overall): # Set ylim only if valid distances exist
        ax2.set_ylim(bottom=max(min_dist_overall / 10, 1e-10))
    if max_iter_plotted_dist > 0:
         ax2.set_xlim(left=-0.05*max_iter_plotted_dist, right=max_iter_plotted_dist*1.05)
    ax2.grid(True, which="both", ls=":", alpha=0.6)

    # Common legend below the plots
    handles, labels = ax2.get_legend_handles_labels()
    if handles: # Only show legend if there are plots
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize='medium', borderaxespad=0.1)

    fig.suptitle("NumPy Optimizers Convergence on 2D Rosenbrock Function", fontsize=14)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout for suptitle and legend

    # Save the figure
    full_filename = os.path.join(output_dir, filename)
    print(f"Saving plot to {full_filename}...")
    try:
        fig.savefig(full_filename, dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Error saving plot {full_filename}: {e}")

    plt.show()


def plot_quadratic_paths_3d(results_np, initial_point, target_minimum=(1, 2, 3), filename="quadratic_paths_3d.png"):
    """Visualizes optimization paths for the 3D quadratic function."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    markers = ['o', 's', 'p', '*', '+', 'x', 'D', 'v', '^', '<', '>']
    idx = 0

    # Determine plot limits based on paths and target
    all_paths_flat = np.concatenate([data['path'] for data in results_np.values() if len(data['path']) > 0])
    min_coords = np.array(target_minimum) - 1
    max_coords = np.array(target_minimum) + 1
    if len(all_paths_flat) > 0:
        valid_paths = all_paths_flat[np.all(np.isfinite(all_paths_flat), axis=1)]
        if len(valid_paths) > 0:
            min_coords = np.minimum(min_coords, valid_paths.min(axis=0) - 0.5)
            max_coords = np.maximum(max_coords, valid_paths.max(axis=0) + 0.5)
    # Ensure start point is within limits
    if np.all(np.isfinite(initial_point)):
        min_coords = np.minimum(min_coords, initial_point - 0.5)
        max_coords = np.maximum(max_coords, initial_point + 0.5)


    for name, data in results_np.items():
        if not len(data['path']): continue
        path = np.array(data['path'])
        valid_path = path[np.all(np.isfinite(path), axis=1)]
        if len(valid_path) == 0: continue

        markevery = max(1, len(valid_path) // 10)
        ax.plot(valid_path[:, 0], valid_path[:, 1], valid_path[:, 2],
                marker=markers[idx % len(markers)], markevery=markevery, markersize=4,
                linestyle='--', label=f"{name} ({data['time']:.2f}s)", alpha=0.7, linewidth=1.5)
        # Mark start and end
        ax.plot([valid_path[0, 0]], [valid_path[0, 1]], [valid_path[0, 2]], 'go', markersize=6, label='_nolegend_') # Start
        if len(valid_path) > 1:
             ax.plot([valid_path[-1, 0]], [valid_path[-1, 1]], [valid_path[-1, 2]], 'ro', markersize=6, label='_nolegend_') # End
        idx += 1

    ax.plot([target_minimum[0]], [target_minimum[1]], [target_minimum[2]], 'kx', markersize=12, mew=3, label=f'Minimum {target_minimum}')
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    start_point_str = f"({initial_point[0]:.2f}, {initial_point[1]:.2f}, {initial_point[2]:.2f})" if np.all(np.isfinite(initial_point)) else "[NaN/Inf]"
    ax.set_title(f"Optimization Paths on 3D Quadratic Function\nStart: {start_point_str}")
    ax.set_xlim(min_coords[0], max_coords[0])
    ax.set_ylim(min_coords[1], max_coords[1])
    ax.set_zlim(min_coords[2], max_coords[2])

    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., fontsize='medium')
    fig.tight_layout(rect=[0, 0, 0.85, 0.96]) # Adjust layout

    # Save the figure
    full_filename = os.path.join(output_dir, filename)
    print(f"Saving plot to {full_filename}...")
    try:
        fig.savefig(full_filename, dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Error saving plot {full_filename}: {e}")

    plt.show()


def plot_quadratic_convergence(results_np, target_minimum=(1, 2, 3), filename="quadratic_convergence.png"):
    """Visualizes convergence for the 3D quadratic function (Loss and Distance)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    target_minimum = np.array(target_minimum, dtype=np.float64)

    # Plot 1: Function Value (Loss) vs Iteration
    ax1 = axes[0]
    min_loss_overall = float('inf')
    max_iter_plotted = 0
    for name, data in results_np.items():
         if not len(data['losses']): continue
         losses = np.array(data['losses'])
         valid_indices = np.isfinite(losses)
         valid_losses = losses[valid_indices]
         valid_iterations = np.arange(len(losses))[valid_indices]
         if len(valid_losses) == 0: continue
         max_iter_plotted = max(max_iter_plotted, len(losses))
         plot_losses = np.maximum(valid_losses, 1e-12)
         min_loss_overall = min(min_loss_overall, plot_losses.min())
         ax1.plot(valid_iterations, plot_losses, label=f"{name}", alpha=0.8)

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Function Value (Loss) f(x)")
    ax1.set_title("Quadratic Convergence: Function Value")
    ax1.set_yscale('log')
    if np.isfinite(min_loss_overall):
        ax1.set_ylim(bottom=max(min_loss_overall / 10, 1e-10))
    if max_iter_plotted > 0:
         ax1.set_xlim(left=-0.05*max_iter_plotted, right=max_iter_plotted*1.05)
    ax1.grid(True, which="both", ls=":", alpha=0.6)

    # Plot 2: Distance to Minimum vs Iteration
    ax2 = axes[1]
    min_dist_overall = float('inf')
    max_iter_plotted_dist = 0
    for name, data in results_np.items():
         if not len(data['path']): continue
         path = np.array(data['path'])
         valid_indices = np.all(np.isfinite(path), axis=1)
         valid_path = path[valid_indices]
         valid_iterations = np.arange(len(path))[valid_indices]
         if len(valid_path) == 0: continue
         max_iter_plotted_dist = max(max_iter_plotted_dist, len(path))

         distances = np.linalg.norm(valid_path - target_minimum, axis=1)
         plot_distances = np.maximum(distances, 1e-12)
         min_dist_overall = min(min_dist_overall, plot_distances.min())
         ax2.plot(valid_iterations, plot_distances, label=f"{name} ({data['time']:.2f}s)", alpha=0.8)

    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Distance to Minimum ||x - x*||")
    ax2.set_title("Quadratic Convergence: Distance to Minimum")
    ax2.set_yscale('log')
    if np.isfinite(min_dist_overall):
         ax2.set_ylim(bottom=max(min_dist_overall / 10, 1e-10))
    if max_iter_plotted_dist > 0:
         ax2.set_xlim(left=-0.05*max_iter_plotted_dist, right=max_iter_plotted_dist*1.05)
    ax2.grid(True, which="both", ls=":", alpha=0.6)

    # Common legend below the plots
    handles, labels = ax2.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize='medium', borderaxespad=0.1)

    fig.suptitle("NumPy Optimizers Convergence on 3D Quadratic Function", fontsize=14)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout

    # Save the figure
    full_filename = os.path.join(output_dir, filename)
    print(f"Saving plot to {full_filename}...")
    try:
        fig.savefig(full_filename, dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Error saving plot {full_filename}: {e}")

    plt.show()


# --- Main Execution ---
if __name__ == "__main__":

    # --- Configuration ---
    optimizers_list_np = [
        "SGD without momentum (NumPy)",
        "SGD with momentum (NumPy)",
        "NAG (NumPy)",
        "Adagrad (NumPy)",
        "Adadelta (NumPy)",
        "RMSprop (NumPy)",
        "Adam (NumPy)",
        "AdaMax (NumPy)",
        "Nadam (NumPy)"
    ]

    # Rosenbrock Configuration
    print("="*30 + "\nOptimizing Rosenbrock Function (2D)\n" + "="*30)
    initial_point_rosenbrock = generate_initial_point(2, low=-2, high=2)
    iterations_rosenbrock = 15000 # Increased iterations
    noise_rosenbrock = 0.0
    print_freq_rosenbrock = 1500 # Print less often
    # Tuned LRs for Rosenbrock (may still need adjustments based on random start)
    specific_lrs_rosenbrock = {
        "SGD without momentum (NumPy)": 5e-6,
        "SGD with momentum (NumPy)": 5e-6,
        "NAG (NumPy)": 5e-6,
        "Adagrad (NumPy)": 0.02, # Requires smaller LR than quadratic
        "Adadelta (NumPy)": 0.5, # Might need tuning; value scales the update
        "RMSprop (NumPy)": 3e-4, # Requires smaller LR
        "Adam (NumPy)": 8e-4,    # Requires smaller LR
        "AdaMax (NumPy)": 1e-3,  # Often more stable
        "Nadam (NumPy)": 8e-4,   # Similar to Adam
    }
    results_rosenbrock_np = run_numpy_optimizers(
        rosenbrock_np, grad_rosenbrock_np, initial_point_rosenbrock, dim=2,
        optimizers_to_run=optimizers_list_np,
        iterations=iterations_rosenbrock,
        gradient_noise_std=noise_rosenbrock,
        specific_lrs=specific_lrs_rosenbrock,
        print_freq=print_freq_rosenbrock
    )

    # Quadratic Configuration
    print("\n" + "="*30 + "\nOptimizing Quadratic Function (3D)\n" + "="*30)
    initial_point_quadratic = generate_initial_point(3, low=-4, high=4)
    iterations_quadratic = 500
    noise_quadratic = 0.0
    print_freq_quadratic = 50
    # LRs for Quadratic (generally more robust)
    specific_lrs_quadratic = {
        "SGD without momentum (NumPy)": 0.05,
        "SGD with momentum (NumPy)": 0.05,
        "NAG (NumPy)": 0.05,
        "Adagrad (NumPy)": 0.3,
        "Adadelta (NumPy)": 1.0, # Scales update
        "RMSprop (NumPy)": 0.03,
        "Adam (NumPy)": 0.05,
        "AdaMax (NumPy)": 0.05,
        "Nadam (NumPy)": 0.05,
    }
    results_quadratic_np = run_numpy_optimizers(
        quadratic_np, grad_quadratic_np, initial_point_quadratic, dim=3,
        optimizers_to_run=optimizers_list_np,
        iterations=iterations_quadratic,
        gradient_noise_std=noise_quadratic,
        specific_lrs=specific_lrs_quadratic,
        print_freq=print_freq_quadratic
    )

    # --- Visualization ---
    print("\n" + "="*30 + "\nGenerating and Saving Plots\n" + "="*30)

    # --- Rosenbrock Plots ---
    print("\n--- Rosenbrock Plots ---")
    plot_rosenbrock_paths(results_rosenbrock_np, initial_point_rosenbrock, target_minimum=(1,1), filename="rosenbrock_2d_paths.png")
    plot_rosenbrock_convergence(results_rosenbrock_np, target_minimum=(1, 1), filename="rosenbrock_2d_convergence.png")

    # --- Quadratic Plots ---
    print("\n--- Quadratic Plots ---")
    plot_quadratic_paths_3d(results_quadratic_np, initial_point_quadratic, target_minimum=(1, 2, 3), filename="quadratic_3d_paths.png")
    plot_quadratic_convergence(results_quadratic_np, target_minimum=(1, 2, 3), filename="quadratic_3d_convergence.png")


    print("\nScript finished. Plots saved in directory:", output_dir)