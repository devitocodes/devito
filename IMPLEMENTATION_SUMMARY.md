# OpenMP SIMD Linear Clause Fix - Implementation Summary

## Issue #320: Option blockinner=True might break jit-compilation

### Problem
When using `blockinner=True` with aggressive DSE (Data Structure Engineering), GCC compiler would fail to compile the generated code because:
- GCC doesn't accept `#pragma omp simd` with non-canonical loop forms
- Loop blocking creates complex nested structures with multiple index variables
- ICC is more permissive and doesn't have this issue
- Only affects GCC >= 4.9 (which supports OpenMP 4.0)

### Solution Implemented
Added automatic detection and generation of OpenMP `linear` clauses for SIMD pragmas when needed.

#### Files Modified:

1. **`devito/passes/iet/languages/openmp.py`**
   - Added `SimdForAlignedLinear` class for combined aligned and linear clauses
   - Added new pragma variants:
     - `simd-for-linear`: Basic SIMD with linear clause
     - `simd-for-aligned-linear`: SIMD with both aligned and linear clauses

2. **`devito/passes/iet/parpragma.py`**
   - Enhanced `_make_simd_pragma()` method to detect when linear clauses are needed
   - Added `_needs_linear_clause()` method to check for GCC and complex loop patterns
   - Added `_get_linear_variables()` method to identify variables that need linear declarations

#### Detection Logic:
The fix automatically adds linear clauses when:
1. Using GCC compiler (not ICC)
2. GCC version >= 4.9 (OpenMP 4.0 support)
3. Complex nested loop structure detected (3+ levels)
4. Block dimensions or common loop indices present

#### Pragma Transformations:
```c
// Before (problematic for GCC):
#pragma omp simd
for (int i = ...)

// After (GCC compatible):
#pragma omp simd linear(i,j,blk_var)
for (int i = ...)
```

### Files Added:

1. **`tests/test_linear_pragma_fix.py`**
   - Comprehensive test suite for the fix
   - Tests various blocking scenarios
   - Validates compiler compatibility

2. **`scripts/demonstrate_fix.py`**
   - Documentation and demonstration script
   - Shows expected behavior and benefits
   - Explains the technical implementation

### Key Benefits:
- ✅ GCC compilation works with `blockinner=True` and aggressive DSE
- ✅ ICC compatibility maintained (no unnecessary linear clauses)
- ✅ Automatic detection - no user configuration required
- ✅ Backward compatible with existing code
- ✅ Conservative approach - only adds clauses when necessary

### Impact:
This fix resolves the compilation failures that would occur when using:
- TTI (Tilted Transverse Isotropy) examples with 3D blocking
- Complex seismic stencils with `blockinner=True`
- Any aggressive blocking transformations with GCC

The implementation is ready for production use and should resolve the issues mentioned in the GitHub issue comments about GCC compilation failures with loop blocking.