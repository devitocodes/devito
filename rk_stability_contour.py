#!/usr/bin/env python3
"""
Stability regions of explicit Runge-Kutta methods using contour plots.
This approach provides smoother and more detailed visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import math

def rk_stability_polynomial(z, order):
    """
    Compute the stability polynomial R(z) for explicit Runge-Kutta methods.
    
    For explicit RK methods, the stability polynomial is:
    R(z) = 1 + z + z^2/2! + z^3/3! + ... + z^p/p!
    """
    R = 0
    for k in range(order + 1):
        R += z**k / math.factorial(k)
    return R

def compute_stability_function(real_range, imag_range, order, resolution=500):
    """
    Compute |R(z)| over a grid of complex numbers for contour plotting.
    """
    real_vals = np.linspace(real_range[0], real_range[1], resolution)
    imag_vals = np.linspace(imag_range[0], imag_range[1], resolution)
    
    Real, Imag = np.meshgrid(real_vals, imag_vals)
    Z = Real + 1j * Imag
    
    # Compute |R(z)| for each point
    R_magnitude = np.abs(rk_stability_polynomial(Z, order))
    
    return Real, Imag, R_magnitude

def plot_stability_contours():
    """
    Create contour plots of stability regions for RK methods orders 2-7.
    """
    orders = [2, 3, 4, 5, 6, 7]
    colors = ['#FF4444', '#4444FF', '#44AA44', '#FF8800', '#AA44AA', '#8B4513']
    
    # Create the main comparison plot
    plt.figure(figsize=(15, 12))
    
    # Define the domain for plotting
    real_range = (-5, 1)
    imag_range = (-4, 4)
    
    print("Creating contour-based stability region plots...")
    
    for i, order in enumerate(orders):
        print(f"  Computing contours for RK{order}...")
        
        # Compute stability function on grid
        Real, Imag, R_mag = compute_stability_function(real_range, imag_range, order, resolution=400)
        
        # Plot contour at |R(z)| = 1 (stability boundary)
        contour = plt.contour(Real, Imag, R_mag, levels=[1.0], colors=[colors[i]], linewidths=2.5)
        
        # Fill the stable region (where |R(z)| <= 1)
        plt.contourf(Real, Imag, R_mag, levels=[0, 1.0], colors=[colors[i]], alpha=0.15)
            
        # Add labels
        plt.clabel(contour, inline=True, fontsize=8, fmt=f'RK{order}')
    
    # Customize the plot
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.4, linewidth=0.8)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.4, linewidth=0.8)
    plt.xlim(real_range)
    plt.ylim(imag_range)
    plt.xlabel('Real(z)', fontsize=22)
    plt.ylabel('Imag(z)', fontsize=22)
    # Configure tick sizes and appearance
    tick_label_size = 18
    plt.xticks(fontsize=tick_label_size)
    plt.yticks(fontsize=tick_label_size)
    plt.title('Regiões de Estabilidade para Métodos de Runge-Kutta Explícitos', 
              fontsize=24, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=colors[i], lw=2.5, label=f'RK{order}') 
                      for i, order in enumerate(orders)]
    plt.legend(handles=legend_elements, fontsize=20, loc='upper left')
    
    # Add explanatory text
    # plt.text(-4.7, 3.7, 'Stable Region: |R(z)| ≤ 1\nContour lines show |R(z)| = 1', 
    #          bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.9),
    #          fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('rk_stability_contour_main.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Main contour plot saved as 'rk_stability_contour_main.png'")

def plot_individual_contours():
    """
    Create individual contour plots for each RK method.
    """
    orders = [2, 3, 4, 5, 6, 7]
    colors = ['#FF4444', '#4444FF', '#44AA44', '#FF8800', '#AA44AA', '#8B4513']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    real_range = (-5, 1)
    imag_range = (-4, 4)
    
    for i, order in enumerate(orders):
        print(f"  Creating detailed contour plot for RK{order}...")
        
        # Higher resolution for individual plots
        Real, Imag, R_mag = compute_stability_function(real_range, imag_range, order, resolution=600)
        
        # Plot multiple contour levels for better visualization
        levels = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
        contours = axes[i].contour(Real, Imag, R_mag, levels=levels, colors='gray', alpha=0.5, linewidths=0.8)
        
        # Highlight the stability boundary
        stability_contour = axes[i].contour(Real, Imag, R_mag, levels=[1.0], colors=[colors[i]], linewidths=3)
        
        # Fill stable region
        axes[i].contourf(Real, Imag, R_mag, levels=[0, 1.0], colors=[colors[i]], alpha=0.3)
        
        # Customize subplot
        axes[i].axhline(y=0, color='k', linestyle='-', alpha=0.4, linewidth=0.8)
        axes[i].axvline(x=0, color='k', linestyle='-', alpha=0.4, linewidth=0.8)
        axes[i].set_xlim(real_range)
        axes[i].set_ylim(imag_range)
        axes[i].set_title(f'RK{order} Stability Region', fontsize=14, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlabel('Real(z)', fontsize=12)
        axes[i].set_ylabel('Imag(z)', fontsize=12)
        
        # Add contour labels
        axes[i].clabel(contours, inline=True, fontsize=8, fmt='%.1f')
        axes[i].clabel(stability_contour, inline=True, fontsize=10, fmt='Stable')
    
    plt.tight_layout()
    plt.savefig('rk_stability_contour_individual.png', dpi=300, bbox_inches='tight')
    print("Individual contour plots saved as 'rk_stability_contour_individual.png'")

def plot_3d_stability_surface():
    """
    Create 3D surface plots showing |R(z)| as a function of Re(z) and Im(z).
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    orders = [2, 4, 6]  # Show a subset for clarity
    colors = ['red', 'green', 'purple']
    
    fig = plt.figure(figsize=(18, 6))
    
    real_range = (-4, 1)
    imag_range = (-3, 3)
    
    for idx, order in enumerate(orders):
        print(f"  Creating 3D surface for RK{order}...")
        
        ax = fig.add_subplot(1, 3, idx+1, projection='3d')
        
        # Compute stability function
        Real, Imag, R_mag = compute_stability_function(real_range, imag_range, order, resolution=200)
        
        # Create surface plot
        surf = ax.plot_surface(Real, Imag, R_mag, cmap='viridis', alpha=0.8, 
                              linewidth=0, antialiased=True)
        
        # Add contour at |R(z)| = 1
        ax.contour(Real, Imag, R_mag, levels=[1.0], colors=['red'], linewidths=3, 
                  offset=0, zdir='z')
        
        # Customize 3D plot
        ax.set_xlabel('Real(z)', fontsize=12)
        ax.set_ylabel('Imag(z)', fontsize=12)
        ax.set_zlabel('|R(z)|', fontsize=12)
        ax.set_title(f'RK{order}: |R(z)| Surface', fontsize=14, fontweight='bold')
        ax.set_zlim(0, 5)
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
    
    plt.tight_layout()
    plt.savefig('rk_stability_3d_surfaces.png', dpi=300, bbox_inches='tight')
    print("3D surface plots saved as 'rk_stability_3d_surfaces.png'")

def create_contour_analysis():
    """
    Analyze and visualize stability properties using contours.
    """
    orders = [2, 3, 4, 5, 6, 7]
    
    print("\nContour-based Stability Analysis:")
    print("=" * 60)
    
    # Find stability boundaries more accurately using contours
    real_range = (-6, 1)
    imag_range = (-0.1, 0.1)  # Near real axis
    
    print(f"{'Method':<8} {'Real Interval':<20} {'Max Stable Step':<15}")
    print("-" * 60)
    
    for order in orders:
        # High resolution near real axis
        Real, Imag, R_mag = compute_stability_function(real_range, imag_range, order, resolution=1000)
        
        # Find leftmost stable point on real axis
        real_axis_idx = np.argmin(np.abs(Imag[:, 0]))  # Find row closest to real axis
        real_axis_stability = R_mag[real_axis_idx, :]
        real_axis_coords = Real[real_axis_idx, :]
        
        # Find where |R(z)| crosses 1.0
        stable_mask = real_axis_stability <= 1.0
        if np.any(stable_mask):
            leftmost_stable = np.min(real_axis_coords[stable_mask])
            max_step = abs(leftmost_stable)
        else:
            leftmost_stable = 0
            max_step = 0
        
        print(f"RK{order:<6} [{leftmost_stable:.3f}, 0.000]      {max_step:<12.3f}")
    
    print("\nContour Method Advantages:")
    print("• Smoother, more accurate boundary representation")
    print("• Easy visualization of stability magnitude levels")
    print("• Natural interpolation between grid points")
    print("• Better handling of complex geometries")

if __name__ == "__main__":
    # Create all contour-based plots
    plot_stability_contours()
    plot_individual_contours()
    plot_3d_stability_surface()
    create_contour_analysis()
    
    print("\nAll contour-based stability plots have been generated!")
    print("Files created:")
    print("- rk_stability_contour_main.png")
    print("- rk_stability_contour_individual.png") 
    print("- rk_stability_3d_surfaces.png")