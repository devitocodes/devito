#!/usr/bin/env python3
"""
Test script to reproduce the blockinner=True issue with OpenMP SIMD pragmas.

This test reproduces the issue mentioned in #320 where using blockinner=True
with aggressive DSE might break JIT compilation when the backend compiler 
is GCC, as GCC doesn't like `#pragma omp simd` if the following loop has 
more than one index variable.
"""

import os
import tempfile
import numpy as np
import sys

# Add devito to path
sys.path.insert(0, '/home/runner/work/devito/devito')

try:
    from devito import Grid, TimeFunction, Eq, Operator, configuration
    from devito.logger import info, set_log_level
    from devito.arch.compiler import GNUCompiler
    
    print("Successfully imported Devito modules")
    
    def test_blockinner_simd_issue():
        """Test case to reproduce the OpenMP SIMD issue with blockinner=True"""
        
        # Set up 3D grid for TTI-like stencil
        shape = (8, 8, 8)  # Small grid for testing
        grid = Grid(shape=shape, dtype=np.float32)
        
        # Create time functions
        u = TimeFunction(name='u', grid=grid, time_order=2, space_order=4)
        v = TimeFunction(name='v', grid=grid, time_order=2, space_order=4)
        
        # Simple stencil equations
        eq1 = Eq(u.forward, u.dt2 + u.laplace)
        eq2 = Eq(v.forward, v.dt2 + v.laplace)
        
        print("Testing with blockinner=True and aggressive DSE...")
        
        # Test with aggressive DSE and blockinner=True
        # This should trigger the issue with GCC
        try:
            op = Operator([eq1, eq2], opt=('advanced', {'blockinner': True}))
            print(f"Generated operator with {len(op._func_table)} compiled functions")
            
            # Check if we can see the generated C code
            if hasattr(op, '_output'):
                code = str(op._output)
                print("Checking for SIMD pragmas in generated code...")
                if '#pragma omp simd' in code:
                    print("Found OpenMP SIMD pragmas")
                    # Look for multiple loop variables
                    lines = code.split('\n')
                    for i, line in enumerate(lines):
                        if '#pragma omp simd' in line and i + 1 < len(lines):
                            next_line = lines[i + 1]
                            print(f"SIMD pragma: {line.strip()}")
                            print(f"Next line: {next_line.strip()}")
                            
                            # Check if this is a potential problem case
                            if 'for' in next_line and any(var in next_line for var in ['i', 'j', 'k']):
                                print("Potential canonical form issue detected")
                else:
                    print("No OpenMP SIMD pragmas found")
            
            # Try to compile and run a simple test
            u.data[:] = 1.0
            v.data[:] = 2.0
            
            print("Testing operator execution...")
            op.apply(time_M=1)
            print("Operator executed successfully")
            
            return True
            
        except Exception as e:
            print(f"Error with blockinner=True: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def check_compiler_info():
        """Check current compiler configuration"""
        print("=== Compiler Information ===")
        print(f"Configuration compiler: {configuration['compiler']}")
        
        # Check if we're using GCC
        try:
            from devito.arch import compiler_registry
            compiler_cls = compiler_registry[configuration['compiler']]
            compiler = compiler_cls()
            print(f"Compiler class: {type(compiler).__name__}")
            print(f"Compiler version: {compiler.version}")
            
            if isinstance(compiler, GNUCompiler):
                print("Using GCC - this test is relevant for the issue")
                if compiler.version >= configuration.get('openmp-version', '4.0'):
                    print("GCC version supports OpenMP 4.0+")
                else:
                    print("GCC version may not support OpenMP 4.0")
            else:
                print("Not using GCC - issue may not occur")
                
        except Exception as e:
            print(f"Could not determine compiler info: {e}")
    
    if __name__ == "__main__":
        print("Testing blockinner=True OpenMP SIMD issue reproduction")
        print("=" * 60)
        
        # Set verbose logging
        set_log_level('DEBUG')
        
        # Check compiler
        check_compiler_info()
        print()
        
        # Run the test
        success = test_blockinner_simd_issue()
        
        if success:
            print("\n✓ Test completed - no immediate compilation errors")
        else:
            print("\n✗ Test failed - reproduced the issue")
        
        print("\nNote: The actual issue may only manifest with specific")
        print("compiler versions and more complex stencils.")

except ImportError as e:
    print(f"Could not import Devito: {e}")
    print("Make sure all dependencies are installed")
    sys.exit(1)