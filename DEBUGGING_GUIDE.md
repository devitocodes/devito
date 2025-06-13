# VS Code Debugging Guide for Devito

## Method 1: Using VS Code Debugger with Breakpoints

### Step 1: Set Up Your Debug Environment
1. Open your Python file (e.g., `debug_basic_example.py`)
2. Ensure you have the Python extension installed in VS Code

### Step 2: Set Breakpoints
- Click in the left margin next to any line number
- A red dot appears = breakpoint is set
- Click the red dot again to remove the breakpoint

### Step 3: Start Debugging
- Press `F5` or go to **Run → Start Debugging**
- Select the Python debugger configuration
- Or use the debug panel (Ctrl+Shift+D)

### Step 4: Debug Controls
When the debugger hits a breakpoint:
- **F10** - Step Over (execute current line)
- **F11** - Step Into (enter function calls)  
- **Shift+F11** - Step Out (exit current function)
- **F5** - Continue (run to next breakpoint)
- **Shift+F5** - Stop debugging

### Step 5: Inspect Variables
When paused at a breakpoint, you can:
1. **Variables Panel** (left side): Shows all local/global variables
2. **Watch Panel**: Add expressions to monitor (e.g., `u.data.shape`)
3. **Hover**: Hover over variables in the code to see their values
4. **Debug Console**: Type variable names or expressions at the bottom

## Method 2: Using `breakpoint()` in Code

Add `breakpoint()` anywhere in your Python code:
```python
# Your code here
u = TimeFunction(name="u", grid=grid, time_order=2, space_order=2)
breakpoint()  # Execution will pause here
# More code here
```

When you run the script, it will pause at `breakpoint()` and drop into an interactive debugger.

## Method 3: Using `pdb` Debugger Commands

When paused at a `breakpoint()`, you can use these commands:
- `l` or `list` - Show current code
- `n` or `next` - Execute next line
- `s` or `step` - Step into functions
- `c` or `continue` - Continue execution
- `p variable_name` - Print variable value
- `pp variable_name` - Pretty print variable
- `w` or `where` - Show call stack
- `u` or `up` - Move up in call stack
- `d` or `down` - Move down in call stack
- `q` or `quit` - Quit debugger

## Method 4: Debugging Jupyter Notebooks

### Option A: Use VS Code's Jupyter debugging
1. Open your `.ipynb` file in VS Code
2. Set breakpoints by clicking in the margin of code cells
3. Run the cell with `Ctrl+Enter`
4. The debugger will pause at breakpoints

### Option B: Use `%debug` magic command
In a Jupyter cell, after an error occurs:
```python
%debug
```
This will start the debugger at the point where the error occurred.

### Option C: Use `%pdb` magic command
At the start of your notebook:
```python
%pdb on
```
This will automatically start the debugger when an exception occurs.

## Useful Debugging Tips for Devito

### 1. Inspect Grid Properties
```python
breakpoint()
print(f"Grid shape: {grid.shape}")
print(f"Grid dimensions: {grid.dimensions}")  
print(f"Grid spacing: {grid.spacing}")
print(f"Grid extent: {grid.extent}")
```

### 2. Inspect Function Data
```python
breakpoint()
print(f"Function data shape: {u.data.shape}")
print(f"Function data type: {u.data.dtype}")
print(f"Min/Max values: {u.data.min():.6f} / {u.data.max():.6f}")
print(f"Function space order: {u.space_order}")
print(f"Function time order: {u.time_order}")
```

### 3. Inspect Operators
```python
breakpoint()
print(f"Operator: {op}")
print(f"Operator body:\n{op.ccode}")  # See generated C code
```

### 4. Monitor Performance
```python
import time
start_time = time.time()
breakpoint()  # Check variables before execution
op(time=10, dt=0.001)
breakpoint()  # Check results after execution
end_time = time.time()
print(f"Execution time: {end_time - start_time:.4f} seconds")
```

## Debugging Common Devito Issues

### Issue 1: Array Shape Mismatches
```python
breakpoint()
print(f"Expected shape: {expected_shape}")
print(f"Actual shape: {actual_array.shape}")
print(f"Array strides: {actual_array.strides}")
```

### Issue 2: Boundary Conditions
```python
breakpoint()
print(f"Boundary data: {u.data[:, 0]}")  # Check boundary values
print(f"Interior data sample: {u.data[50:55, 50:55]}")  # Check interior
```

### Issue 3: Time Stepping Issues
```python
breakpoint()
print(f"Current time index: {u._time_position}")
print(f"Time buffer content: {u.data}")
```