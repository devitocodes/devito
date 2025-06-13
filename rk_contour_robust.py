#!/usr/bin/env python3
"""
Robust contour-based stability regions for explicit Runge-Kutta methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import math

def rk_stability_polynomial(z, order):
    """Compute the stability polynomial R(z) for explicit RK methods."""
    R = 0
    for k in range(order + 1):
        R += z**k / math.factorial(k)
    return R

def compute_stability_grid(order, resolution=400):
    """Compute |R(z)| over a grid for contour plotting."""
    # Define the complex plane domain
    real_vals = np.linspace(-5, 1, resolution)
    imag_vals = np.linspace(-4, 4, resolution)
    
    Real, Imag = np.meshgrid(real_vals, imag_vals)
    Z = Real + 1j * Imag
    
    # Compute |R(z)| for each point
    R_magnitude = np.abs(rk_stability_polynomial(Z, order))
    
    return Real, Imag, R_magnitude

def create_contour_plots():
    """Create clean contour plots for RK stability regions."""
    orders = [2, 3, 4, 5, 6, 7]
    colors = ['#FF4444', '#4444FF', '#44AA44', '#FF8800', '#AA44AA', '#8B4513']
    
    # Main plot with all methods
    plt.figure(figsize=(14, 10))
    
    print("Creating contour-based stability plots...")
    
    for i, order in enumerate(orders):
        print(f"  Processing RK{order}...")
        
        Real, Imag, R_mag = compute_stability_grid(order, resolution=300)
        
        # Plot stability boundary (|R(z)| = 1)
        contour_lines = plt.contour(Real, Imag, R_mag, levels=[1.0], 
                                   colors=[colors[i]], linewidths=2.5)
        
        # Fill stable region (|R(z)| <= 1)
        plt.contourf(Real, Imag, R_mag, levels=[0, 1.0], 
                    colors=[colors[i]], alpha=0.2)
    
    # Formatting
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.4, linewidth=0.8)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.4, linewidth=0.8)
    plt.xlim(-5, 1)
    plt.ylim(-4, 4)
    plt.xlabel('Real(z)', fontsize=14)
    plt.ylabel('Imag(z)', fontsize=14)
    plt.title('Stability Regions - Contour Method\nExplicit Runge-Kutta Methods (Orders 2-7)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    
    # Manual legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], alpha=0.5, label=f'RK{order}') 
                      for i, order in enumerate(orders)]
    plt.legend(handles=legend_elements, fontsize=12, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('rk_contour_main.png', dpi=300, bbox_inches='tight')
    print("Main contour plot saved as 'rk_contour_main.png'")
    
    # Individual detailed plots
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for i, order in enumerate(orders):
        print(f"  Creating detailed plot for RK{order}...")
        
        Real, Imag, R_mag = compute_stability_grid(order, resolution=400)
        
        # Multiple contour levels for detail
        levels = np.array([0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0])
        
        # Background contours
        axes[i].contour(Real, Imag, R_mag, levels=levels, 
                       colors='gray', alpha=0.4, linewidths=0.8)
        
        # Stability boundary
        boundary = axes[i].contour(Real, Imag, R_mag, levels=[1.0], 
                                  colors=[colors[i]], linewidths=3)
        
        # Stable region fill
        axes[i].contourf(Real, Imag, R_mag, levels=[0, 1.0], 
                        colors=[colors[i]], alpha=0.3)
        
        # Formatting
        axes[i].axhline(y=0, color='k', linestyle='-', alpha=0.4)
        axes[i].axvline(x=0, color='k', linestyle='-', alpha=0.4)
        axes[i].set_xlim(-5, 1)
        axes[i].set_ylim(-4, 4)
        axes[i].set_title(f'RK{order} Stability Region', fontsize=13, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlabel('Real(z)')
        axes[i].set_ylabel('Imag(z)')
    
    plt.tight_layout()
    plt.savefig('rk_contour_individual.png', dpi=300, bbox_inches='tight')
    print("Individual contour plots saved as 'rk_contour_individual.png'")

def create_filled_contour_plot():
    """Create a filled contour plot showing stability magnitude levels."""
    orders = [2, 4, 6]  # Subset for clarity
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, order in enumerate(orders):
        print(f"  Creating filled contour for RK{order}...")
        
        Real, Imag, R_mag = compute_stability_grid(order, resolution=500)
        
        # Create filled contour plot
        levels = np.logspace(-1, 1, 20)  # Logarithmic levels from 0.1 to 10
        
        contourf = axes[idx].contourf(Real, Imag, R_mag, levels=levels, 
                                     cmap='RdYlBu_r', extend='max')
        
        # Add stability boundary
        axes[idx].contour(Real, Imag, R_mag, levels=[1.0], 
                         colors=['black'], linewidths=3)
        
        # Formatting
        axes[idx].axhline(y=0, color='white', linestyle='-', alpha=0.8)
        axes[idx].axvline(x=0, color='white', linestyle='-', alpha=0.8)
        axes[idx].set_xlim(-5, 1)
        axes[idx].set_ylim(-4, 4)
        axes[idx].set_title(f'RK{order}: |R(z)| Magnitude', fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Real(z)')
        axes[idx].set_ylabel('Imag(z)')
        
        # Add colorbar
        cbar = plt.colorbar(contourf, ax=axes[idx])
        cbar.set_label('|R(z)|', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('rk_contour_filled.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Filled contour plots saved as 'rk_contour_filled.png'")

def analyze_contour_stability():
    """Analyze stability using contour method."""
    orders = [2, 3, 4, 5, 6, 7]
    
    print("\n" + "="*60)
    print("CONTOUR-BASED STABILITY ANALYSIS")
    print("="*60)
    
    print(f"{'Method':<8} {'Real Interval':<20} {'Area (approx.)':<12}")
    print("-" * 50)
    
    for order in orders:
        # Get stability data
        Real, Imag, R_mag = compute_stability_grid(order, resolution=600)
        
        # Find real axis stability
        mid_row = R_mag.shape[0] // 2  # Middle row (imaginary ≈ 0)
        real_axis_stability = R_mag[mid_row, :]
        real_coords = Real[mid_row, :]
        
        # Find stability interval on real axis
        stable_indices = real_axis_stability <= 1.0
        if np.any(stable_indices):
            stable_reals = real_coords[stable_indices]
            left_bound = np.min(stable_reals)
            right_bound = np.max(stable_reals)
        else:
            left_bound = right_bound = 0
        
        # Approximate area of stability region
        stable_region = R_mag <= 1.0
        grid_area = (Real[0, 1] - Real[0, 0]) * (Imag[1, 0] - Imag[0, 0])
        approx_area = np.sum(stable_region) * grid_area
        
        print(f"RK{order:<6} [{left_bound:.3f}, {right_bound:.3f}]      {approx_area:<10.2f}")
    
    print("\nContour Method Benefits:")
    print("• Smooth, continuous boundary representation")
    print("• Natural visualization of |R(z)| magnitude")
    print("• Accurate identification of stability regions")
    print("• Easy to see complex geometry details")
    print("• Efficient computation on regular grids")

if __name__ == "__main__":
    create_contour_plots()
    create_filled_contour_plot()
    analyze_contour_stability()
    
    print("\n" + "="*60)
    print("CONTOUR-BASED PLOTS COMPLETED")
    print("="*60)
    print("Generated files:")
    print("• rk_contour_main.png - Main comparison plot")
    print("• rk_contour_individual.png - Individual detailed plots")
    print("• rk_contour_filled.png - Filled contour magnitude plots")
    print("\nThese plots use matplotlib's contour functionality for")
    print("smooth, accurate representation of stability boundaries.")