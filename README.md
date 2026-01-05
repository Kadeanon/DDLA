# DDLA - Double Dense Linear Algebra

DDLA is a CPU double-precision dense linear algebra library written in pure C#, focused on numerical correctness, computational performance, and scalability.

## Features

### Core Design

- **Flexible memory layout**: Uses a managed array–backed, freely strided layout with internal optimizations for classic row-major and column-major storage. This makes it easy to interoperate with other array-based linear algebra code.
- **Layered architecture**: Follows the classic BLAS structure, with a set of pluggable execution kernels as the driver layer, and higher-level matrix algorithms built on top.
- **Algorithm-oriented struct type**: Algorithmic kernels operate on small, immutable struct views layered beneath the top-level container types. These share the same logical representation while avoiding extra heap allocations and reducing GC pressure.
- **SIMD and parallel acceleration**: Performance-critical operations make heavy use of `System.Numerics.Vector<double>` for SIMD vectorization, and top-level BLAS operations are parallelized using the Task Parallel Library (TPL).

### Data Structures

- **Matrix / MatrixView**: Matrix container class and immutable struct supporting row-major, column-major, and arbitrary stride layouts.
- **Vector / VectorView**: Vector container class and immutable struct with arbitrary stride support.
- **MArray**: Tensor support (work in progress).

## Current Status

### Implemented Functionality

#### BLAS Operations

- **Level 1**: Vector operations (dot product, scaling, copy, etc.)
- **Level 2**: Matrix–vector operations (GeMV, GeR, SyMV, etc.)
- **Level 3**: Matrix–matrix operations (GeMM, SyRk, TrSM, etc.)
- A small number of non-computational helper routines rely on the built-in `blis` library; all three levels of the standard BLAS interfaces (Level 1–3) have pure managed implementations.

#### Matrix Decompositions

- **LU decomposition**: LU decomposition for square matrices, with partial pivoting.
- **Cholesky decomposition**: Cholesky decomposition for symmetric positive-definite matrices.
- **LDLT decomposition**: LDLT decomposition for symmetric matrices.
- **QR decomposition**: QR decomposition based on Householder transformations.
- **Symmetric eigenvalue decomposition**: Symmetric EVD using QR iteration.
- **Singular value decomposition**: SVD using QR iteration.

#### Matrix Transformations

- **Householder transformations**: Used in QR decomposition and tridiagonalization.
- **Tridiagonalization**: Tridiagonal reduction for symmetric matrices.
- **Bidiagonalization**: Bidiagonal reduction for general matrices.

### Planned Features

- Faster EVD / symmetric SVD implementations and other advanced linear algebra methods.
- A basic tensor library built on top of the linear algebra core.
- Double-precision complex support, including interoperability between real and complex structures.
- ...

## Quick Start

```csharp
using DDLA.Core;
using DDLA.BLAS.Managed;

// Build a simple 2x2 matrix and a vector, compute y := A * x and print the result.

// Construct a 2x2 matrix from an array, default row-major layout, equivalent to {{1, 2}, {3, 4}}.
var A = new Matrix([1, 2, 3, 4], 2, 2);
// Construct a length-2 vector filled with ones.
var x = Vector.Create(2, 1);
// Perform the matrix–vector multiply.
var y = A * x;
// Print vector y; 
// output: [3, 7]
Console.WriteLine(y);
```

## Goals and Non-Goals

DDLA originates from my quantum chemistry computing projects, so its primary target is traditional scientific computing rather than machine learning.

### Goals

- **Pure managed linear algebra**  
  DDLA aims to be a purely managed linear algebra library. Compared with bindings to highly optimized native libraries, you should expect some performance gap at the extreme high end, which is an intentional and acceptable trade-off.

- **Prioritize double-precision real and complex**  
  Providing equally high-quality implementations for many numeric types adds a lot of complexity. DDLA will first focus on double-precision real types, then extend to double-precision complex. Until mixed real/complex operations are well supported, other numeric types will not be a priority.

- **Scalable multithreading**  
  Multithreading is already enabled in managed BLAS Level-3 routines, but the current implementation is still rough. Improving scalability and scheduling strategies for parallel execution is an ongoing focus.

- **Tensor support for scientific computing**  
  Tensor operations are a major next step, primarily targeting scientific computing workloads (especially electronic structure calculations), rather than deep learning workloads.

### Non-Goals / Lower-Priority Directions

- **Sparse matrices and lazy containers**  
  Due to the heavy use of pointer-style arithmetic on managed references, the current internal representation is better suited for dense, explicitly stored matrices. Traditional sparse matrix containers and “compute-on-read” lazy containers are costly to support within this architecture and are not intended to be primary goals.

- **Automatic differentiation and general ML frameworks**  
  While tensor support and some advanced linear algebra routines are planned, DDLA does not plan to implement an automatic differentiation system, nor to provide direct integration with general-purpose machine learning frameworks.

- **Short-term GPU support**  
  GPU support is considered a long-term direction, but there is still substantial CPU-side work to be done first. In the near term, DDLA will not invest heavily in GPU backends.
