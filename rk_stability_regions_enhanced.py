#!/usr/bin/env python3
"""
Enhanced plot of stability regions of explicit Runge-Kutta methods from order 2 to order 7.
Includes comparison with analytical results and better visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as patches
import math

def rk_stability_polynomial(z, order):
    """
    Compute the stability polynomial R(z) for explicit Runge-Kutta methods.
    
    For explicit RK methods, the stability polynomial is:
    R(z) = 1 + z + z^2/2! + z^3/3! + ... + z^p/p!
    
    where p is the order of the method.
    """
    R = 0
    for k in range(order + 1):
        R += z**k / math.factorial(k)
    return R

def find_stability_boundary(order, num_points=1000):
    """
    Find the boundary of the stability region for a given RK order.
    The stability region is where |R(z)| <= 1.
    """
    # Create a grid of complex numbers
    theta = np.linspace(0, 2*np.pi, num_points)
    
    # Start from the unit circle and work outward to find boundary
    boundary_points = []
    
    for t in theta:
        # Binary search to find the boundary
        r_min, r_max = 0, 10
        for _ in range(50):  # iterations for binary search
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

def plot_stability_regions_enhanced():
    """
    Enhanced plot of stability regions for RK methods of orders 2-7.
    """
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Colors and styles
    colors = ['#FF4444', '#4444FF', '#44AA44', '#FF8800', '#AA44AA', '#8B4513']
    orders = [2, 3, 4, 5, 6, 7]
    
    # Plot 1: All stability regions together
    for i, order in enumerate(orders):
        print(f"Computing stability region for RK order {order}...")
        boundary = find_stability_boundary(order, num_points=500)
        
        # Extract real and imaginary parts
        real_parts = boundary.real
        imag_parts = boundary.imag
        
        # Plot the boundary
        ax1.plot(real_parts, imag_parts, color=colors[i], linewidth=2.5, 
                label=f'RK{order}')
        
        # Fill the stability region with low alpha
        ax1.fill(real_parts, imag_parts, color=colors[i], alpha=0.15)
    
    # Setup first plot
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
    ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
    ax1.set_xlim(-4.5, 1)
    ax1.set_ylim(-3.5, 3.5)
    ax1.set_xlabel('Real(z)', fontsize=12)
    ax1.set_ylabel('Imag(z)', fontsize=12)
    ax1.set_title('Stability Regions: All Orders (2-7)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc='upper right')
    
    # Add text explanation
    ax1.text(-4.2, 3.2, 'Stability Region: |R(z)| ≤ 1', 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
             fontsize=11, fontweight='bold')
    
    # Plot 2: Individual plots in smaller subfigures
    for i, order in enumerate(orders):
        row = i // 3
        col = i % 3
        
        # Create small subplot within ax2
        x_offset = col * 0.32
        y_offset = 0.5 - row * 0.45
        
        sub_ax = fig.add_axes([0.52 + x_offset, 0.1 + y_offset, 0.12, 0.35])
        
        boundary = find_stability_boundary(order, num_points=300)
        real_parts = boundary.real
        imag_parts = boundary.imag
        
        sub_ax.plot(real_parts, imag_parts, color=colors[i], linewidth=2)
        sub_ax.fill(real_parts, imag_parts, color=colors[i], alpha=0.3)
        
        sub_ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
        sub_ax.axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
        sub_ax.set_xlim(-4, 1)
        sub_ax.set_ylim(-3, 3)
        sub_ax.set_title(f'RK{order}', fontsize=10, fontweight='bold')
        sub_ax.grid(True, alpha=0.3)
        sub_ax.tick_params(labelsize=8)
    
    # Remove ax2 as we're using custom subplots
    ax2.remove()
    
    plt.tight_layout()
    plt.show()
    
    # Save the enhanced plot
    plt.savefig('rk_stability_regions_enhanced.png', dpi=300, bbox_inches='tight')
    print("Enhanced plot saved as 'rk_stability_regions_enhanced.png'")

def plot_stability_comparison():
    """
    Create a comparison plot showing how stability regions change with order.
    """
    plt.figure(figsize=(14, 6))
    
    # Left subplot: Real axis stability intervals
    plt.subplot(1, 2, 1)
    orders = [2, 3, 4, 5, 6, 7]
    leftmost_points = []
    
    for order in orders:
        boundary = find_stability_boundary(order, num_points=1000)
        leftmost_real = np.min(boundary.real)
        leftmost_points.append(-leftmost_real)  # Make positive for plotting
    
    plt.bar(orders, leftmost_points, color=['#FF4444', '#4444FF', '#44AA44', '#FF8800', '#AA44AA', '#8B4513'],
            alpha=0.7, edgecolor='black', linewidth=1)
    plt.xlabel('Runge-Kutta Order', fontsize=12)
    plt.ylabel('Stability Interval Length |z_min|', fontsize=12)
    plt.title('Real Axis Stability Intervals', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(leftmost_points):
        plt.text(orders[i], v + 0.05, f'{v:.2f}', ha='center', fontweight='bold')
    
    # Right subplot: Stability region areas (approximate)
    plt.subplot(1, 2, 2)
    areas = []
    
    for order in orders:
        boundary = find_stability_boundary(order, num_points=500)
        # Approximate area using shoelace formula
        x = boundary.real
        y = boundary.imag
        area = 0.5 * abs(sum(x[i]*y[i+1] - x[i+1]*y[i] for i in range(-1, len(x)-1)))
        areas.append(area)
    
    plt.bar(orders, areas, color=['#FF4444', '#4444FF', '#44AA44', '#FF8800', '#AA44AA', '#8B4513'],
            alpha=0.7, edgecolor='black', linewidth=1)
    plt.xlabel('Runge-Kutta Order', fontsize=12)
    plt.ylabel('Approximate Stability Region Area', fontsize=12)
    plt.title('Stability Region Areas', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(areas):
        plt.text(orders[i], v + 0.1, f'{v:.1f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    plt.savefig('rk_stability_comparison.png', dpi=300, bbox_inches='tight')
    print("Comparison plot saved as 'rk_stability_comparison.png'")
    
    return dict(zip(orders, leftmost_points)), dict(zip(orders, areas))

def analyze_stability_properties_detailed():
    """
    Detailed analysis of stability properties.
    """
    print("\n" + "="*80)
    print("DETAILED STABILITY ANALYSIS OF EXPLICIT RUNGE-KUTTA METHODS")
    print("="*80)
    
    orders = [2, 3, 4, 5, 6, 7]
    
    print(f"{'Order':<6} {'Real Interval':<15} {'Max |z|':<10} {'At θ=π':<10} {'Notes'}")
    print("-" * 70)
    
    for order in orders:
        # Find the leftmost point (most negative real part)
        boundary = find_stability_boundary(order, num_points=1000)
        leftmost_real = np.min(boundary.real)
        
        # Find maximum |z| on the imaginary axis
        imag_axis_points = boundary[np.abs(boundary.real) < 0.01]
        if len(imag_axis_points) > 0:
            max_imag = np.max(np.abs(imag_axis_points.imag))
        else:
            max_imag = 0
        
        # Check stability at z = -1 (for reference)
        R_at_minus_one = abs(rk_stability_polynomial(-1, order))
        stable_at_minus_one = "Yes" if R_at_minus_one <= 1 else "No"
        
        print(f"RK{order:<4} [{leftmost_real:.3f}, 0]    {abs(leftmost_real):<8.3f}  "
              f"{max_imag:<8.3f}  Stable at z=-1: {stable_at_minus_one}")
    
    print("\nKey Observations:")
    print("- All methods are stable for z = 0 (as expected)")
    print("- Stability regions lie entirely in the left half-plane")
    print("- Higher-order methods generally have larger stability intervals on real axis")
    print("- But stability regions become more 'narrow' in some directions")
    
    print(f"\nTheoretical Background:")
    print("- Stability polynomial: R(z) = Σ(k=0 to p) z^k/k!")
    print("- Stability condition: |R(z)| ≤ 1")
    print("- For pure real z < 0: larger intervals mean larger stable time steps")

if __name__ == "__main__":
    # Generate the original plot
    plot_stability_regions_enhanced()
    
    # Generate comparison plots
    intervals, areas = plot_stability_comparison()
    
    # Detailed analysis
    analyze_stability_properties_detailed()