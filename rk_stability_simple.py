#!/usr/bin/env python3
"""
Simple plot of stability regions of explicit Runge-Kutta methods from order 2 to order 7.
Saves plots without displaying them interactively.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import math

def rk_stability_polynomial(z, order):
    """
    Compute the stability polynomial R(z) for explicit Runge-Kutta methods.
    """
    R = 0
    for k in range(order + 1):
        R += z**k / math.factorial(k)
    return R

def find_stability_boundary(order, num_points=1000):
    """
    Find the boundary of the stability region for a given RK order.
    """
    theta = np.linspace(0, 2*np.pi, num_points)
    boundary_points = []
    
    for t in theta:
        # Binary search to find the boundary
        r_min, r_max = 0, 10
        for _ in range(50):
            r = (r_min + r_max) / 2
            z = r * np.exp(1j * t)
            R = rk_stability_polynomial(z, order)
            
            if abs(R) > 1:
                r_max = r
            else:
                r_min = r
        
        z_boundary = r_min * np.exp(1j * t)
        boundary_points.append(z_boundary)
    
    return np.array(boundary_points)

def create_stability_plots():
    """
    Create and save stability region plots.
    """
    plt.style.use('default')
    
    # Main plot with all stability regions
    plt.figure(figsize=(12, 10))
    
    colors = ['#FF4444', '#4444FF', '#44AA44', '#FF8800', '#AA44AA', '#8B4513']
    orders = [2, 3, 4, 5, 6, 7]
    
    print("Creating stability regions plot...")
    
    for i, order in enumerate(orders):
        print(f"  Computing RK{order} stability region...")
        boundary = find_stability_boundary(order, num_points=500)
        
        real_parts = boundary.real
        imag_parts = boundary.imag
        
        plt.plot(real_parts, imag_parts, color=colors[i], linewidth=2.5, 
                label=f'RK{order}', zorder=10-i)
        plt.fill(real_parts, imag_parts, color=colors[i], alpha=0.15, zorder=10-i)
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.4, linewidth=0.8)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.4, linewidth=0.8)
    plt.xlim(-4.5, 1)
    plt.ylim(-4, 4)
    plt.xlabel('Real(z)', fontsize=14)
    plt.ylabel('Imag(z)', fontsize=14)
    plt.title('Stability Regions of Explicit Runge-Kutta Methods (Orders 2-7)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='upper right')
    
    # Add explanatory text
    plt.text(-4.2, 3.6, 'Stability Region: |R(z)| ≤ 1\nwhere R(z) = Σ(k=0 to p) z^k/k!', 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.9),
             fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('rk_stability_main.png', dpi=300, bbox_inches='tight')
    print("Main plot saved as 'rk_stability_main.png'")
    
    # Individual plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, order in enumerate(orders):
        print(f"  Creating individual plot for RK{order}...")
        boundary = find_stability_boundary(order, num_points=300)
        
        axes[i].plot(boundary.real, boundary.imag, color=colors[i], linewidth=2.5)
        axes[i].fill(boundary.real, boundary.imag, color=colors[i], alpha=0.3)
        axes[i].axhline(y=0, color='k', linestyle='-', alpha=0.4)
        axes[i].axvline(x=0, color='k', linestyle='-', alpha=0.4)
        axes[i].set_xlim(-4.5, 1)
        axes[i].set_ylim(-3.5, 3.5)
        axes[i].set_title(f'RK{order} Stability Region', fontsize=12, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlabel('Real(z)')
        axes[i].set_ylabel('Imag(z)')
    
    plt.tight_layout()
    plt.savefig('rk_stability_individual.png', dpi=300, bbox_inches='tight')
    print("Individual plots saved as 'rk_stability_individual.png'")
    
    # Analysis table
    print("\nStability Analysis Results:")
    print("=" * 60)
    print(f"{'Method':<8} {'Real Interval':<15} {'Interval Length':<15}")
    print("-" * 60)
    
    for order in orders:
        boundary = find_stability_boundary(order, num_points=1000)
        leftmost = np.min(boundary.real)
        print(f"RK{order:<6} [{leftmost:.3f}, 0.000]     {abs(leftmost):<12.3f}")
    
    print("\nKey Properties:")
    print("• All stability regions lie in the left half-plane (Re(z) ≤ 0)")
    print("• Higher-order methods have larger stability intervals on the real axis")
    print("• Stability regions become more complex for higher orders")
    print("• The method is stable if the scaled time step λΔt lies within the region")
    print("  where λ is the eigenvalue of the spatial discretization operator")

if __name__ == "__main__":
    create_stability_plots()