#!/usr/bin/env python3
"""
Plot stability regions of explicit Runge-Kutta methods from order 2 to order 7.
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

def plot_stability_regions():
    """
    Plot stability regions for RK methods of orders 2-7.
    """
    plt.figure(figsize=(12, 10))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    orders = [2, 3, 4, 5, 6, 7]
    
    for i, order in enumerate(orders):
        print(f"Computing stability region for RK order {order}...")
        boundary = find_stability_boundary(order, num_points=500)
        
        # Extract real and imaginary parts
        real_parts = boundary.real
        imag_parts = boundary.imag
        
        # Plot the boundary
        plt.plot(real_parts, imag_parts, color=colors[i], linewidth=2, 
                label=f'RK{order}')
        
        # Fill the stability region with low alpha
        plt.fill(real_parts, imag_parts, color=colors[i], alpha=0.1)
    
    # Add coordinate axes
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Set up the plot
    plt.xlim(-4, 2)
    plt.ylim(-4, 4)
    plt.xlabel('Real(z)', fontsize=12)
    plt.ylabel('Imag(z)', fontsize=12)
    plt.title('Stability Regions of Explicit Runge-Kutta Methods (Orders 2-7)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Add text explanation
    plt.text(-3.5, 3.5, 'Stability Region: |R(z)| ≤ 1', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Also save the plot
    plt.savefig('rk_stability_regions.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'rk_stability_regions.png'")

def analyze_stability_properties():
    """
    Analyze and print some properties of the stability regions.
    """
    print("\nStability Properties of Explicit Runge-Kutta Methods:")
    print("=" * 60)
    
    orders = [2, 3, 4, 5, 6, 7]
    
    for order in orders:
        # Find the leftmost point (most negative real part)
        boundary = find_stability_boundary(order, num_points=1000)
        leftmost_real = np.min(boundary.real)
        
        print(f"RK{order}: Approximate stability interval on real axis: "
              f"[{leftmost_real:.3f}, 0]")
    
    print("\nNote:")
    print("- Larger stability regions allow larger time steps")
    print("- Higher-order methods generally have smaller stability regions")
    print("- The stability region shrinks as the order increases")

if __name__ == "__main__":
    plot_stability_regions()
    analyze_stability_properties()