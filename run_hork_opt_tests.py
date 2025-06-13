#!/usr/bin/env python3
"""
Manual test runner for Test_HORK_OPT class to bypass pytest configuration issues.
"""

import os
import sys

# Set environment variables to help with Devito configuration
os.environ['DEVITO_ARCH'] = 'gcc'
os.environ['DEVITO_PLATFORM'] = 'cpu64'
os.environ['DEVITO_LOGGING'] = 'WARNING'

# Add current directory to Python path
sys.path.insert(0, '.')

try:
    print("Attempting to import Devito components...")
    
    # Try importing step by step to identify where it fails
    import numpy as np
    print("✓ NumPy imported successfully")
    
    import sympy as sym
    print("✓ SymPy imported successfully")
    
    # Try importing Devito
    from devito import Grid, Function, TimeFunction
    print("✓ Basic Devito components imported successfully")
    
    from devito import Derivative, Operator, solve, Eq, configuration
    print("✓ Advanced Devito components imported successfully")
    
    from devito.types.multistage_new import multistage_method, MultiStage
    print("✓ Multistage components imported successfully")
    
    # Import the test class
    from tests.test_multistage_new import Test_HORK_OPT, grid_parameters, time_parameters
    print("✓ Test class imported successfully")
    
    # Create test instance and run tests
    test_instance = Test_HORK_OPT()
    
    print("\n" + "="*60)
    print("Running Test_HORK_OPT.test_coupled_op_computing_exp...")
    print("="*60)
    
    try:
        test_instance.test_coupled_op_computing_exp()
        print("✓ test_coupled_op_computing_exp PASSED")
    except Exception as e:
        print(f"✗ test_coupled_op_computing_exp FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("Running Test_HORK_OPT.test_HORK_EXP_convergence...")
    print("="*60)
    
    # Run convergence test for a few degrees (3, 4, 5 to avoid long runtime)
    degrees_to_test = [3, 4, 5]
    
    for degree in degrees_to_test:
        try:
            print(f"\nTesting degree {degree}...")
            test_instance.test_HORK_EXP_convergence(degree)
            print(f"✓ test_HORK_EXP_convergence(degree={degree}) PASSED")
        except Exception as e:
            print(f"✗ test_HORK_EXP_convergence(degree={degree}) FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("Test execution completed!")
    print("="*60)
    
except ImportError as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
    
except Exception as e:
    print(f"Unexpected error: {e}")
    import traceback
    traceback.print_exc()