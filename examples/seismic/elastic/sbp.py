"""
Staggered summation-by-parts (SBP) difference and interpolation operators, fourth
order in the interior and second order at a boundary, of

    O. O'Reilly and N. A. Petersson, Energy conservative SBP discretizations of
    the acoustic wave equation in covariant form on staggered curvilinear grids,
    Journal of Computational Physics 411 (2020), 109386.

The boundary closures are those of github.com/ooreilly/sbp (covariant/resources/
build3), MIT License, Copyright (c) 2017 ooreilly: Permission is hereby granted,
free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and
to permit persons to whom the Software is furnished to do so, subject to the
following conditions: The above copyright notice and this permission notice
shall be included in all copies or substantial portions of the Software. THE
SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

The operators act on two grids of the half line `r >= 0` with spacing `h`, the
nodes `r_i = i h`, i >= 0, the `+` grid, and the half points with the boundary,
`r_0 = 0` and `r_j = (j - 1/2) h`, j >= 1, the `-` grid. `D+` and `P+` take a
function on the `-` grid to the `+` grid, `D-` and `P-` the other way. With the
diagonal quadrature weights `H+` and `H-`, they satisfy the summation-by-parts
property

    H- D- = -(H+ D+)^T + B^T,    H+ P+ = (H- P-)^T,

where `B` picks the boundary values, `B[0, 0] = -1`, so that a discretization
pairing them conserves energy up to the boundary terms.
"""

import numpy as np

__all__ = ['sbp_closure_rows', 'sbp_interior', 'sbp_matrix', 'sbp_norm']

# Interior stencils, from the `-` grid to the `+` grid and the other way: the
# offsets of the points read from the point computed, in the indices of each grid
_INTERIOR = {
    'D': (np.array([1/24, -9/8, 9/8, -1/24]), np.array([-1, 0, 1, 2])),
    'P': (np.array([-1/16, 9/16, 9/16, -1/16]), np.array([-1, 0, 1, 2])),
}

# Boundary closures: rows 0 to 3 of each operator, columns from the boundary
_CLOSURES = {
    ('D', '+'): np.array([
        [-1.7779989465546748, 1.3337480247900155, 0.7775013168066564,
         -0.33325039504199694, 0., 0.],
        [-0.44102173413920587, -0.17308424848898904, 0.4487228323259927,
         0.1653831503022022, 0., 0.],
        [0.17987932138827012, -0.27572572541507884, -0.9597948548284453,
         1.1171892610431817, -0.06154800218792766, 0.],
        [0.015391138150708755, 0.05688514555035906, -0.19989764645971708,
         -0.8628231468598346, 1.0285385292191949, -0.03809401960071092]]),
    ('D', '-'): np.array([
        [-1.4511412472637157, 1.853423741791147, -0.3534237417911469,
         -0.04885875273628437, 0.],
        [-0.8577143189081458, 0.5731429567244373, 0.4268570432755628,
         -0.14228568109185424, 0.],
        [-0.16745485058828774, -0.4976354482351368, 0.4976354482351368,
         0.16745485058828774, 0.],
        [0.10270611134051243, -0.26245413264698597, -0.8288742701021167,
         1.0342864927831414, -0.04566420137455131]]),
    ('P', '+'): np.array([
        [1.021634663287431, 0.2819848069393812, -0.6180562720973399,
         0.31443680187052775, 0., 0.],
        [0.042239016673327356, 0.3396410147129031, 0.7151204288908755,
         -0.09700046027710588, 0., 0.],
        [0.07303948488311952, -0.1894963474067583, 0.6040719793238264,
         0.6047068864817039, -0.09232200328189148, 0.],
        [-0.048777201969098134, -0.005056639538068934, 0.032043489806347844,
         0.5646621164922883, 0.5142692646095974, -0.05714102940106638]]),
    ('P', '-'): np.array([
        [0.8338228784688313, 0.17751233164292596, 0.14350670130765417,
         -0.15484191141941145, 0.],
        [0.1813404047323969, 1.1246711188154426, -0.2933634518280757,
         -0.012648071719763686, 0.],
        [-0.13311427062823994, 0.7930714675884345, 0.31319987670785077,
         0.026842926331954644, 0.],
        [0.09690785566330458, -0.1539344946680898, 0.44864912028443893,
         0.6768738207821733, -0.06849630206182696]]),
}

# Quadrature weights of the first points of each grid, one elsewhere
_NORMS = {
    '+': np.array([0.2812150147607664, 1.4480216223843674, 0.6769783776156325,
                   1.0937849852392336]),
    '-': np.array([0.344556397209992, 0.43729008856459833, 1.30569549651249,
                   0.9124580177129197]),
}


def sbp_interior(kind):
    """
    The interior stencil of the operator `kind` ('D' or 'P'), the fourth-order
    staggered derivative or interpolation, as its weights, in units of the
    spacing, and the offsets of the points it reads from the point it computes,
    in the indices of the grid it reads, from the `-` grid to the `+` grid.
    """
    weights, offsets = _INTERIOR[kind]
    return tuple(float(w) for w in weights), tuple(int(o) for o in offsets)


def sbp_closure_rows():
    """The number of boundary rows of the operators, before their interior stencils."""
    return max(len(closure) for closure in _CLOSURES.values())


def sbp_norm(grid, size):
    """
    The quadrature weights of the first `size` points of the `grid` ('+' or '-'),
    in units of the spacing.
    """
    weights = np.ones(size)
    boundary = _NORMS[grid][:size]
    weights[:len(boundary)] = boundary
    return weights


def sbp_matrix(kind, grid, size, traction_free=False):
    """
    The first `size` rows of the operator `kind` ('D' or 'P') onto the `grid`
    ('+' or '-'), as a dense matrix with the columns of the points of the other
    grid it reads, in units of the spacing.

    With `traction_free`, the difference operator is the one of the momentum
    equation with a weakly imposed zero boundary flux, `D - H^-1 B`, which drops
    the flux on the boundary from the derivative (O'Reilly et al., 2021, A
    high-order finite-difference method on staggered curvilinear grids for
    seismic wave propagation applications with topography, BSSA 112).
    """
    weights, offsets = _INTERIOR[kind]
    closure = _CLOSURES[(kind, grid)]
    # From the `-` grid, the point i of the `+` grid reads the half points i - 1/2
    # and above, at indices i and above; from the `+` grid, the point j of the
    # `-` grid reads the nodes j - 1 and above
    shift = 0 if grid == '+' else -1
    matrix = np.zeros((size, size + 4))
    for row in range(size):
        if row < len(closure):
            matrix[row, :closure.shape[1]] = closure[row]
        else:
            matrix[row, row + shift + offsets] = weights
    if traction_free:
        if kind != 'D':
            raise ValueError("Only a difference operator drops the boundary flux")
        matrix[0, 0] += 1/_NORMS[grid][0]
    return matrix
