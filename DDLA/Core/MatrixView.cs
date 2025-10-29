using DDLA.BLAS;
using DDLA.Misc;
using DDLA.Misc.Flags;
using DDLA.Misc.Pools;
using DDLA.UFuncs;
using DDLA.UFuncs.Operators;
using System.Collections;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

using BlasProvider = DDLA.BLAS.Managed.BlasProvider;

namespace DDLA.Core;

/// <summary>
/// A span of a MatrixView with a reference to the first element.
/// </summary>
/// <typeparam name="double">The type of the elements in the MatrixView.</typeparam>
[DebuggerDisplay("{(long)Rows}x{(long)Cols}")]
[DebuggerTypeProxy(typeof(MatrixSpanDebugView))]
public readonly struct MatrixView : IEnumerable<double>
{
    #region Properties
    internal double[] Data { get; }
    internal int Offset { get; }
    public int Rows { get; }
    public int Cols { get; }
    public int RowStride { get; }
    public int ColStride { get; }
    public int DiagOffset { get; }

    public readonly int MinDim => Math.Min(Rows, Cols);

    public readonly int MaxDim => Math.Max(Rows, Cols);

    public readonly SingleIndice RowIndice => new(Rows, RowStride);

    public readonly SingleIndice ColIndice => new(Cols, ColStride);

    public readonly bool IsEmpty => Rows == 0 || Cols == 0;

    public readonly int Size => Cols * Rows;
    #endregion Properties

    #region Constructors
    public MatrixView(double[] array)
    {
        ArgumentNullException.ThrowIfNull(array, nameof(array));
        Data = array;
        Offset = 0;
        int length = array.Length;
        Rows = length;
        Cols = 1;
        RowStride = 1;
        ColStride = 1;
        DiagOffset = 0;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public MatrixView(double[] array, int rows, int cols, bool colMajor = false)
    {
        ArgumentNullException.ThrowIfNull(array, nameof(array));
        ArgumentOutOfRangeException.ThrowIfNegative(rows, nameof(rows));
        ArgumentOutOfRangeException.ThrowIfNegative(cols, nameof(cols));
        int length = rows * cols;
        ArgumentOutOfRangeException.ThrowIfLessThan(array.Length, length, nameof(array));
        Data = array;
        Offset = 0;
        Rows = rows;
        Cols = cols;
        if (colMajor)
        {
            RowStride = 1;
            ColStride = rows;
        }
        else
        {
            RowStride = cols;
            ColStride = 1;
        }
        DiagOffset = 0;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public MatrixView(double[] array, int offset, int rows, int cols, bool colMajor = false)
    {
        ArgumentNullException.ThrowIfNull(array, nameof(array));
        ArgumentOutOfRangeException.ThrowIfNegative(offset, nameof(offset));
        ArgumentOutOfRangeException.ThrowIfNegative(rows, nameof(rows));
        ArgumentOutOfRangeException.ThrowIfNegative(cols, nameof(cols));
        int length = rows * cols;
        ArgumentOutOfRangeException.ThrowIfLessThan(array.Length, offset + length,
            nameof(array));
        Data = array;
        Offset = offset;
        Rows = rows;
        Cols = cols;
        if (colMajor)
        {
            RowStride = 1;
            ColStride = rows;
        }
        else
        {
            RowStride = cols;
            ColStride = 1;
        }
        DiagOffset = 0;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public MatrixView(double[] array, int offset, int rows, int cols,
        int rowStride, int colStride, int diagOffset = 0)
    {
        ArgumentNullException.ThrowIfNull(array, nameof(array));
        ArgumentOutOfRangeException.ThrowIfNegative(offset, nameof(offset));
        ArgumentOutOfRangeException.ThrowIfNegative(rows, nameof(rows));
        ArgumentOutOfRangeException.ThrowIfNegative(cols, nameof(cols));
        int last = offset + (rows - 1) * rowStride + (cols - 1) * colStride + 1;
        Data = array;
        Offset = offset;
        Rows = rows;
        Cols = cols;
        RowStride = rowStride;
        ColStride = colStride;
        DiagOffset = diagOffset;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public MatrixView(VectorView vector)
    {
        Data = vector.Data;
        Offset = vector.Offset;
        Rows = vector.Length;
        RowStride = vector.Stride;
        Cols = 1;
        ColStride = Rows * vector.Stride;
        DiagOffset = 0;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public MatrixView(VectorView vector, int rows, int cols)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(vector.Length, rows * cols,
            nameof(vector));

        Data = vector.Data;
        Offset = vector.Offset;
        Rows = rows;
        RowStride = vector.Stride;
        Cols = cols;
        ColStride = rows * vector.Stride;
        DiagOffset = 0;
    }
    #endregion Constructors

    #region Alloc
    public static MatrixView Create(int rows, int cols, bool uninited = false, bool colMajor = false)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(rows, nameof(rows));
        ArgumentOutOfRangeException.ThrowIfNegative(cols, nameof(cols));
        int length = rows * cols;
        ArgumentOutOfRangeException.ThrowIfGreaterThan(length, Array.MaxLength, "size");
        double[] data = uninited ? GC.AllocateUninitializedArray<double>(length)
            : new double[length];
        return new Matrix(data, rows, cols, colMajor);
    }

    public static MatrixView Filled(int rows, int cols, double val, bool colMajor = false)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows, nameof(rows));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(cols, nameof(cols));
        int length = rows * cols;
        ArgumentOutOfRangeException.ThrowIfGreaterThan(length, Array.MaxLength, "size");
        double[] data = new double[length];
        Array.Fill(data, val);
        return new Matrix(data, rows, cols, colMajor);
    }

    public static MatrixView Eyes(int length, bool colMajor = false)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length, nameof(length));
        int size = length * length;
        double[] data = new double[size];
        var diagStride = length + 1;
        for (int i = 0; i < length; i++)
        {
            data[i * diagStride] = 1.0;
        }
        return new Matrix(data, length, length, colMajor);
    }

    public static MatrixView Diagonals(VectorView vector, bool colMajor = false)
    {
        if (vector.Length == 0)
            return new(Array.Empty<double>());
        var result = Create(vector.Length, vector.Length, colMajor);
        vector.CopyTo(result.Diag);
        return result;
    }

    public static MatrixView RandomDense(int rows, int cols, Random? random = null)
    {
        random ??= Random.Shared;
        ArgumentOutOfRangeException.
            ThrowIfNegative(rows, nameof(rows));
        ArgumentOutOfRangeException.
            ThrowIfNegative(cols, nameof(cols));

        var mat = Create(rows, cols, uninited: true);
        var data = mat.Data;
        for (int i = 0; i < data.Length; i++)
            data[i] = random.NextDouble();
        return mat;
    }

    public static MatrixView RandomTriangle(int dim,
        UpLo uplo = UpLo.Lower, DiagType diag = DiagType.NonUnit,
        Random? random = null)
    {
        random ??= Random.Shared;
        ArgumentOutOfRangeException.
            ThrowIfNegative(dim, nameof(dim));

        var mat = Create(dim, dim, uninited: true);
        if (uplo == UpLo.Upper)
        {
            for (int i = 0; i < dim; i++)
            {
                double sum = 0;
                var val = random.NextDouble();
                mat[i, i] = val;
                for (int j = i + 1; j < dim; j++)
                {
                    val = random.NextDouble();
                    mat[i, j] = val;
                    sum += val * val;
                    mat[j, i] = 0;
                }
                sum = 1 / Math.Sqrt(sum);
                for (int j = i + 1; j < dim; j++)
                {
                    mat[i, j] *= sum;
                }
                if (diag == DiagType.Unit)
                {
                    mat[i, i] = 1.0;
                }
            }
        }
        else
        {
            for (int i = 0; i < dim; i++)
            {
                double sum = 0;
                var val = random.NextDouble();
                for (int j = 0; j < i; j++)
                {
                    mat[i, j] = val;
                    sum += val * val;
                    mat[j, i] = 0;
                    val = random.NextDouble();
                }
                mat[i, i] = val;
                sum = 1 / Math.Sqrt(sum);
                for (int j = 0; j < i; j++)
                {
                    mat[i, j] *= sum;
                }
                if (diag == DiagType.Unit)
                {
                    mat[i, i] = 1.0;
                }
            }
        }
        return mat;
    }

    public static MatrixView RandomSPD(int dim, Random? random = null)
    {
        random ??= Random.Shared;
        ArgumentOutOfRangeException.
            ThrowIfNegative(dim, nameof(dim));
        var r = RandomTriangle(dim, UpLo.Lower, DiagType.NonUnit, random);
        var mat = r.EmptyLike();
        BlasProvider.SyRk(UpLo.Lower, 1, r, 0, mat);
        BlasProvider.MakeSy(mat, UpLo.Lower);
        return mat;
    }

    public static MatrixView RandomSymmetric(int dim,
        DiagType diag = DiagType.NonUnit, Random? random = null)
    {
        random ??= Random.Shared;
        ArgumentOutOfRangeException.
            ThrowIfNegativeOrZero(dim, nameof(dim));
        var mat = Create(dim, dim, uninited: true);
        for (int i = 0; i < dim; i++)
        {
            if (diag == DiagType.Unit)
            {
                mat[i, i] = 1.0;
            }
            else if (diag == DiagType.NonUnit)
            {
                mat[i, i] = random.NextDouble();
            }
            for (int j = i + 1; j < dim; j++)
            {
                double val = random.NextDouble();
                mat[i, j] = val;
                mat[j, i] = val;
            }
        }
        return mat;
    }

    public MatrixView EmptyLike(bool uninited = false)
        => Create(Rows, Cols, uninited);

    public MatrixView FilledLike(double val)
        => Filled(Rows, Cols, val);
    #endregion Alloc

    #region Accessor and Slicer
    public readonly ref double this[int row, int col]
    {
        get
        {
            if (row < 0) row += Rows;
            if (row < 0 || row >= Rows)
                throw new ArgumentOutOfRangeException(nameof(row), $"Row index {row} is out of range.");
            if (col < 0) col += Cols;
            if (col < 0 || col >= Cols)
                throw new ArgumentOutOfRangeException(nameof(col), $"Column index {col} is out of range.");
            return ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(Data), Offset + row * RowStride + col * ColStride);
        }
    }

    public readonly ref double this[Index rowIndex, Index colIndex]
    {
        get
        {
            var row = rowIndex.GetOffset(Rows);
            if (row < 0 || row >= Rows)
                throw new ArgumentOutOfRangeException($"Row index {row} is out of range.", nameof(row));
            var col = colIndex.GetOffset(Cols);
            if (col < 0 || col >= Cols)
                throw new ArgumentOutOfRangeException($"Column index {col} is out of range.", nameof(col));
            return ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(Data), Offset + row * RowStride + col * ColStride);
        }
    }

    public readonly MatrixView this[Range rows, Range cols]
    {
        get
        {
            (int startRow, int lengthRow) = rows.GetOffsetAndLength(Rows);
            (int startCol, int lengthCol) = cols.GetOffsetAndLength(Cols);
            if (startRow < 0 || lengthRow < 0 || startRow + lengthRow > Rows)
                throw new ArgumentOutOfRangeException(nameof(rows), $"Row range {rows} is out of range.");
            if (startCol < 0 || lengthCol < 0 || startCol + lengthCol > Cols)
                throw new ArgumentOutOfRangeException(nameof(cols), $"Column range {cols} is out of range.");
            return SliceSubUncheck(startRow, lengthRow, startCol, lengthCol);
        }
        set
        {
            (int startRow, int lengthRow) = rows.GetOffsetAndLength(Rows);
            (int startCol, int lengthCol) = cols.GetOffsetAndLength(Cols);
            if (startRow < 0 || lengthRow < 0 || startRow + lengthRow > Rows)
                throw new ArgumentOutOfRangeException(nameof(rows), $"Row range {rows} is out of range.");
            if (startCol < 0 || lengthCol < 0 || startCol + lengthCol > Cols)
                throw new ArgumentOutOfRangeException(nameof(cols), $"Column range {cols} is out of range.");
            value.CopyTo(SliceSubUncheck(startRow, lengthRow, startCol, lengthCol));
        }
    }

    public readonly VectorView this[Index rowIndex, Range cols]
    {
        get
        {
            int row = rowIndex.GetOffset(Rows);
            (int startCol, int lengthCol) = cols.GetOffsetAndLength(Cols);
            if (row < 0 || row >= Rows)
                throw new ArgumentOutOfRangeException(nameof(rowIndex),
                    $"Row index {row} is out of range.");
            if (startCol < 0 || lengthCol < 0 || startCol + lengthCol > Cols)
                throw new ArgumentOutOfRangeException(nameof(cols),
                    $"Column range {cols} is out of range.");
            return SliceRowUncheck(row, startCol, lengthCol);
        }
        set
        {
            int row = rowIndex.GetOffset(Rows);
            (int startCol, int lengthCol) = cols.GetOffsetAndLength(Cols);
            if (row < 0 || row >= Rows)
                throw new ArgumentOutOfRangeException(nameof(rowIndex),
                    $"Row index {row} is out of range.");
            if (startCol < 0 || lengthCol < 0 || startCol + lengthCol > Cols)
                throw new ArgumentOutOfRangeException(nameof(cols),
                    $"Column range {cols} is out of range.");
            value.CopyTo(SliceRowUncheck(row, startCol, lengthCol));
        }
    }

    public readonly VectorView this[Range rows, Index colIndex]
    {
        get
        {
            (int startRow, int lengthRow) = rows.GetOffsetAndLength(Rows);
            int col = colIndex.GetOffset(Cols);
            if (startRow < 0 || lengthRow < 0 || startRow + lengthRow > Rows)
                throw new ArgumentOutOfRangeException(nameof(rows),
                    $"Row range {rows} is out of range.");
            if (col < 0 || col >= Cols)
                throw new ArgumentOutOfRangeException(nameof(colIndex),
                    $"Column index {col} is out of range.");
            return SliceColUncheck(col, startRow, lengthRow);
        }
        set
        {
            (int startRow, int lengthRow) = rows.GetOffsetAndLength(Rows);
            int col = colIndex.GetOffset(Cols);
            if (startRow < 0 || lengthRow < 0 || startRow + lengthRow > Rows)
                throw new ArgumentOutOfRangeException(nameof(rows),
                    $"Row range {rows} is out of range.");
            if (col < 0 || col >= Cols)
                throw new ArgumentOutOfRangeException(nameof(colIndex),
                    $"Column index {col} is out of range.");
            value.CopyTo(SliceColUncheck(col, startRow, lengthRow));
        }
    }

    public readonly ref double GetHeadRef()
    {
        if (IsEmpty)
            return ref Unsafe.NullRef<double>();
        return ref Data[Offset];
    }

    public readonly ref double GetPinnableReference()
        => ref GetHeadRef();

    public readonly VectorView GetRow(int row)
    {
        if (row < 0) row += Rows;
        if (row < 0 || row >= Rows)
            throw new ArgumentOutOfRangeException(nameof(row),
                $"Row index {row} is out of range.");
        return GetRowUncheck(row);
    }

    public readonly VectorView GetColumn(int col)
    {
        if (col < 0) col += Cols;
        if (col < 0 || col >= Cols)
            throw new ArgumentOutOfRangeException(nameof(col),
                $"Column index {col} is out of range.");
        return GetColUncheck(col);
    }

    public readonly VectorView Diag
    {
        get
        {
            int diagLength = Math.Min(Rows, Cols);
            return new(Data, Offset, diagLength, RowStride + ColStride);
        }

        set
        {
            int diagLength = Math.Min(Rows, Cols);
            var target = new VectorView(Data, Offset, diagLength, RowStride + ColStride);
            BlasProvider.Copy(value, target);
        }
    }

    public MatrixView T => new(Data, Offset, Cols, Rows, ColStride, RowStride,-DiagOffset);

    internal readonly ref double AtUncheck(int row, int col)
        => ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(Data),
            Offset + row * RowStride + col * ColStride);

    internal readonly VectorView GetRowUncheck(int row)
        => new(Data, Offset + row * RowStride, Cols, ColStride);

    internal readonly VectorView SliceRowUncheck(int row, int colStart)
        => new(Data, Offset + row * RowStride + colStart * ColStride,
            Cols - colStart, ColStride);

    internal readonly VectorView SliceRowUncheck(int row, int colStart, int colLength)
        => new(Data, Offset + row * RowStride + colStart * ColStride,
            colLength, ColStride);

    internal readonly VectorView GetColUncheck(int col)
        => new(Data, Offset + col * ColStride, Rows, RowStride);

    internal readonly VectorView SliceColUncheck(int col, int rowStart)
        => new(Data, Offset + rowStart * RowStride + col * ColStride,
            Rows - rowStart, RowStride);

    internal readonly VectorView SliceColUncheck(int col, int rowStart, int rowLength)
        => new(Data, Offset + rowStart * RowStride + col * ColStride,
            rowLength, RowStride);

    internal readonly MatrixView SliceSubUncheck(int rowStart, int colStart)
        => new(Data, Offset + rowStart * RowStride + colStart * ColStride,
            Rows - rowStart, Cols - colStart, RowStride, ColStride);

    internal readonly MatrixView SliceSubUncheck(int rowStart, int rowLength, int colStart, int colLength)
        => new(Data, Offset + rowStart * RowStride + colStart * ColStride,
            rowLength, colLength, RowStride, ColStride);
    #endregion Accessor and Slicer

    #region Memory
    public readonly void FlattenTo(Span<double> target)
    {
        if (target.Length < Size)
            throw new ArgumentException($"Target span is too small. " +
                $"Required: {Size}, Actual: {target.Length}");
        ref double targetRef = ref target[0];
        ref double head = ref Data[Offset];


        if (ColStride == 1)
        {
            for (int i = 0; i < Rows; i++)
            {
                MemoryMarshal.CreateSpan(ref head, Cols)
                        .CopyTo(MemoryMarshal.CreateSpan(ref targetRef, Cols));
                head = ref Unsafe.Add(ref head, RowStride);
                targetRef = ref Unsafe.Add(ref targetRef, Cols);
            }
        }
        else
        {
            for (int i = 0; i < Rows; i++)
            {
                ref double ptr = ref head;
                int j = 0;
                for (; j <= Cols - 4; j += 4)
                {
                    targetRef = ptr;
                    Unsafe.Add(ref targetRef, 1) = Unsafe.Add(ref ptr, ColStride);
                    Unsafe.Add(ref targetRef, 2) = Unsafe.Add(ref ptr, ColStride * 2);
                    Unsafe.Add(ref targetRef, 3) = Unsafe.Add(ref ptr, ColStride * 3);
                    ptr = ref Unsafe.Add(ref ptr, ColStride * 4);
                    targetRef = ref Unsafe.Add(ref targetRef, 4);
                }
                for (; j < Cols; j++)
                {
                    targetRef = Unsafe.Add(ref ptr, j * ColStride);
                    ptr = ref Unsafe.Add(ref ptr, ColStride);
                    targetRef = ref Unsafe.Add(ref targetRef, 1);
                }
                head = ref Unsafe.Add(ref head, RowStride);
            }
        }
    }

    public readonly void Fill(double val)
    {
        BlasProvider.Set(val, this);
    }

    public readonly void Fill(double val, UpLo upLo, bool ignoreDiag = false)
    {
        BlasProvider.Set(ignoreDiag ? DiagType.Unit : DiagType.NonUnit, upLo, val, this);
    }

    public readonly void Clear()
    {
        BlasProvider.Set(0.0, this);
    }

    public readonly void Clear(UpLo upLo, bool ignoreDiag = false)
    {
        BlasProvider.Set(ignoreDiag ? DiagType.Unit : DiagType.NonUnit, upLo, 0.0, this);
    }

    public readonly void CopyTo(MatrixView other)
    {
        BlasProvider.Copy(this, other);
    }

    public readonly Matrix Clone(bool colMajor = false)
    {
        var target = colMajor ? Create(Cols, Rows, true).T :
            Create(Rows, Cols, true);
        BlasProvider.Copy(this, target);
        return new(target);
    }

    public readonly void MakeTr(UpLo uplo)
    {
        BlasProvider.MakeTr(this, uplo);
    }

    public readonly void MakeSy(UpLo uplo)
    {
        BlasProvider.MakeSy(this, uplo);
    }

    public readonly VectorView Flatten(bool forceCopy = false)
    {
        if (!forceCopy && RowStride == Cols * ColStride)
        {
            return new VectorView(Data, Offset, Size, ColStride);
        }
        double[] arr = new double[Size];
        FlattenTo(arr);
        return new(arr);
    }

    public readonly MatrixView Expand(int length = 1)
    {
        var result = Matrix.Create(
            Rows + length, Cols + length, uninited: true);
        var slice =
            result[new Range(0, Rows), new Range(0, Cols)];
        CopyTo(slice);
        return result;
    }

    internal readonly Span<double> GetSpan()
    {
        if (IsEmpty)
            throw new InvalidOperationException("Matrix is empty.");
        return Data.AsSpan(Offset);
    }

    /// <summary>
    /// If the MatrixView is not row-major layer,
    /// it will create a new continuous MatrixView.
    /// </summary>
    /// <remarks>Just use for BLAS.</remarks>
    /// <returns>A row-majo</returns>
    public readonly MatrixView MakeRowMajor()
    {
        if (ColStride == 1)
        {
            return this; // Already continuous
        }
        return Clone();
    }
    #endregion Memory

    #region IEnumerable
    readonly IEnumerator<double> IEnumerable<double>.GetEnumerator()
        => new MatrixEnumberator(this);

    readonly IEnumerator IEnumerable.GetEnumerator()
        => new MatrixEnumberator(this);

    public readonly RefMatrixEnumberator GetEnumerator()
        => new(this);

    internal struct MatrixEnumberator : IEnumerator<double>
    {
        private MatrixView _matrix;
        private int _currentRow;
        private int _currentCol;
        public MatrixEnumberator(MatrixView MatrixView)
        {
            _matrix = MatrixView;
            _currentRow = 0;
            _currentCol = 0;
        }

        public readonly double Current
            => _matrix.AtUncheck(_currentRow, _currentCol);

        readonly double IEnumerator<double>.Current => Current;

        readonly object IEnumerator.Current => Current;

        public bool MoveNext()
        {
            if (_currentCol < _matrix.Cols - 1)
            {
                _currentCol++;
                return true;
            }
            else if (_currentRow < _matrix.Rows - 1)
            {
                _currentCol = 0;
                _currentRow++;
                return true;
            }
            else
            {
                return false;
            }
        }

        public void Reset()
        {
            _currentRow = 0;
            _currentCol = 0;
        }

        readonly void IDisposable.Dispose() { }
    }

    public ref struct RefMatrixEnumberator
    {
        private ref double _currentVal;
        private int _currentRow;
        private readonly int _rowEnd;
        private readonly int _rowStride;
        private int _currentCol;
        private readonly int _colEnd;
        private readonly int _colStride;
        private bool _init;
        public RefMatrixEnumberator(MatrixView MatrixView)
        {
            _currentRow = 0;
            _currentCol = 0;
            _rowEnd = MatrixView.Rows - 1;
            _colEnd = MatrixView.Cols - 1;
            _rowStride = MatrixView.RowStride;
            _colStride = MatrixView.ColStride;
            _currentVal = ref MatrixView.AtUncheck(0, 0);
            _init = false;
        }
        public ref double Current
            => ref _currentVal;

        public bool MoveNext()
        {
            if (!_init)
            {
                _init = true;
                return _currentRow <= _rowEnd && _currentCol <= _colEnd;
            }

            if (_currentCol < _colEnd)
            {
                _currentCol++;
                _currentVal = ref Unsafe.Add(ref _currentVal, _colStride);
                return true;
            }
            else if (_currentRow < _rowEnd)
            {
                _currentCol = 0;
                _currentVal = ref Unsafe.Subtract(ref _currentVal, _colEnd * _colStride);
                _currentRow++;
                _currentVal = ref Unsafe.Add(ref _currentVal, _rowStride);
                return true;
            }
            else
                return false;

        }

        public void Reset()
        {
            _currentVal = ref Unsafe.Subtract(ref _currentVal,
                _currentRow * _colStride + _currentCol * _rowStride);
            _currentRow = 0;
            _currentCol = 0;
        }
    }
    #endregion IEnumerable

    #region Math
    public static MatrixView operator +(MatrixView self)
    {
        return self.Clone();
    }

    public static MatrixView operator +(MatrixView left, MatrixView right)
    {
        right = right.Clone();
        BlasProvider.Add(left, right);
        return right;
    }

    public static MatrixView operator +(double left, MatrixView right)
        => right + left;

    public static MatrixView operator +(MatrixView left, double right)
    {
        var res = left.EmptyLike();
        UFunc.Map<AddOperator<double>, double>
            (left, right, res, default);
        return res;
    }

    public static MatrixView operator -(MatrixView self)
    {
        var res = self.EmptyLike();
        UFunc.Map<NegateOperator<double>>
            (self, res, default);
        return res;
    }

    public static MatrixView operator -(MatrixView left, MatrixView right)
    {
        left = left.Clone();
        BlasProvider.Sub(right, left);
        return left;
    }

    public static MatrixView operator -(MatrixView left, double right)
    {
        var res = left.EmptyLike();
        UFunc.Map<SubtractOperator<double>, double>
            (left, right, res, default);
        return res;
    }

    public static MatrixView operator -(double left, MatrixView right)
    {
        var res = right.EmptyLike();
        UFunc.Map<SwappedOp<SubtractOperator<double>, double, double>, double>
            (right, left, res, default);
        return res;
    }

    public static MatrixView operator *(double left, MatrixView right)
    {
        var dest = Matrix.Create(right.Rows, right.Cols, uninited: true);
        BlasProvider.Scal2(left, right, dest);
        return dest;
    }

    public static MatrixView operator *(MatrixView left, double right)
        => right * left;

    public static MatrixView operator /(MatrixView left, double right)
        => left * (1.0 / right);

    public static VectorView operator *(MatrixView left, VectorView right)
    {
        var dest = VectorView.Create(left.Rows, true);
        BlasProvider.GeMV(1.0, left, right, 0.0, dest);
        return dest;
    }

    public static MatrixView operator *(MatrixView left, MatrixView right)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(left.Cols, right.Rows,
            "Matrix multiplication requires the number of columns in the left MatrixView " +
            "to be equal to the number of rows in the right MatrixView.");
        int m = left.Rows;
        int n = right.Cols;
        var dest = Matrix.Create(m, n, uninited: true);
        BlasProvider.GeMM(1.0, left, right, 0.0, dest);
        return dest;
    }

    public readonly MatrixView AddedBy(MatrixView other)
    {
        BlasProvider.Add(other, this);
        return this;
    }

    public readonly MatrixView AddedBy(double alpha, MatrixView other)
    {
        BlasProvider.Axpy(alpha, other, this);
        return this;
    }

    public readonly MatrixView SubtractedBy(MatrixView other)
    {
        BlasProvider.Sub(other, this);
        return this;
    }

    public readonly MatrixView ScaledBy(double alpha)
    {
        BlasProvider.Scal(alpha, this);
        return this;
    }

    public readonly MatrixView InvScaledBy(double alpha)
    {
        BlasProvider.InvScal(alpha, this);
        return this;
    }

    public readonly MatrixView ScaledTo(double alpha, MatrixView? output = null)
    {
        var result = output ?? EmptyLike();
        BlasProvider.Scal(alpha, result);
        return result;
    }

    public readonly MatrixView Multify(MatrixView other, MatrixView? output = null)
        => Multify(1.0, other, output);

    public readonly MatrixView Multify(MatrixView other, double beta, MatrixView output)
        => Multify(1.0, other, beta, output);

    public readonly MatrixView Multify(double alpha, MatrixView other, MatrixView? output = null)
    {
        var m = Rows;
        var k = Cols;
        ArgumentOutOfRangeException.ThrowIfNotEqual(other.Rows, k, nameof(other));
        var n = other.Cols;
        if (output is MatrixView result)
        {
            ArgumentOutOfRangeException.ThrowIfNotEqual(result.Rows, m, nameof(output));
            ArgumentOutOfRangeException.ThrowIfNotEqual(result.Cols, n, nameof(output));
        }
        else
        {
            result = Matrix.Create(m, n);
        }
        BlasProvider.GeMM(alpha, this, other, 0.0, result);
        return result;
    }

    public readonly MatrixView Multify(double alpha, MatrixView other, double beta, MatrixView output)
    {
        var m = Rows;
        var k = Cols;
        ArgumentOutOfRangeException.ThrowIfNotEqual(other.Rows, k, nameof(other));
        var n = other.Cols;
        ArgumentOutOfRangeException.ThrowIfNotEqual(output.Rows, m, nameof(output));
        ArgumentOutOfRangeException.ThrowIfNotEqual(output.Cols, n, nameof(output));
        BlasProvider.GeMM(alpha, this, other, beta, output);
        return output;
    }

    public readonly VectorView Multify(VectorView other, VectorView? output = null)
        => Multify(1.0, other, output);

    public readonly VectorView Multify(VectorView other, double beta, VectorView output)
        => Multify(1.0, other, beta, output);

    public readonly VectorView Multify(double alpha, VectorView other, VectorView? output = null)
    {
        var m = Rows;
        var k = Cols;
        ArgumentOutOfRangeException.ThrowIfNotEqual(other.Length, k, nameof(other));
        if (output is VectorView result)
        {
            ArgumentOutOfRangeException.ThrowIfNotEqual(result.Length, m, nameof(output));
        }
        else
        {
            result = Vector.Create(m);
        }
        BlasProvider.GeMV(alpha, this, other, 0.0, result);
        return result;
    }

    public readonly VectorView Multify(double alpha, VectorView other, double beta, VectorView output)
    {
        var m = Rows;
        var k = Cols;
        ArgumentOutOfRangeException.ThrowIfNotEqual(other.Length, k, nameof(other));
        ArgumentOutOfRangeException.ThrowIfNotEqual(output.Length, m, nameof(output));
        BlasProvider.GeMV(alpha, this, other, beta, output);
        return output;
    }

    public readonly void Rank1(VectorView x, VectorView y)
        => BlasProvider.GeR(1.0, x, y, this);

    public readonly void Rank1(double alpha, VectorView x, VectorView y)
        => BlasProvider.GeR(alpha, x, y, this);

    public readonly void Rank1(UpLo uplo, VectorView x)
        => BlasProvider.SyR(uplo, 1.0, x, this);

    public readonly void Rank1(UpLo uplo, double alpha, VectorView x)
        => BlasProvider.SyR(uplo, alpha, x, this);

    public readonly void Rank2(UpLo uplo, VectorView x, VectorView y)
        => BlasProvider.SyR2(uplo, 1.0, x, y, this);

    public readonly void Rank2(UpLo uplo, double alpha, VectorView x, VectorView y)
        => BlasProvider.SyR2(uplo, alpha, x, y, this);

    public readonly void ShiftDiag(double alpha)
        => BlasProvider.Shift(alpha, Diag); 
    
    public readonly void SwapCol(int i, int j)
        => BlasProvider.Swap(GetColumn(i), GetColumn(j));

    public readonly void SwapRow(int i, int j)
        => BlasProvider.Swap(GetRow(i), GetRow(j));
    #endregion Math

    #region Ufunc
    public readonly MatrixView Pointwise<TUnaryAction>()
        where TUnaryAction : struct, IUnaryOperator<double, double>
    {
        var dest = EmptyLike();
        UFunc.Map<TUnaryAction>(this, dest, new());
        return dest;
    }

    public readonly MatrixView PointwiseAbs()
        => Pointwise<AbsOperator<double>>();

    public readonly double Nrm1() =>
        BlasProvider.Nrm1(this);

    public readonly double NrmF() =>
        BlasProvider.NrmF(this);

    public readonly double NrmInf() =>
        BlasProvider.NrmInf(this);
    #endregion Ufunc

    public readonly override string ToString()
    {
        using var _ = StringBuilderPool.Borrow(out var sb);
        sb.Append('[');
        for (int i = 0; i < Rows; i++)
        {
            GetRow(i).ToString(sb, format: "F6");
            if (i < Rows - 1)
                sb.AppendLine(",");
        }
        sb.AppendLine("]");
        return sb.ToString();
    }

    [DebuggerDisplay("{Rows}x{Cols}")]
    public struct MatrixSpanDebugView
    {
        public object Items;
        public MatrixSpanDebugView(MatrixView mat)
        {
            Rows = mat.Rows;
            Cols = mat.Cols;
            RowStride = mat.RowStride;
            ColStride = mat.ColStride;

            if (mat.IsEmpty || TooLong)
            {
                Items = Array.Empty<double>();
            }
            else
            {
                Dictionary<(long x, long y), double>
                    values = new(mat.Size);
                for (int i = 0; i < mat.Rows; i++)
                {
                    for (int j = 0; j < mat.Cols; j++)
                    {
                        values[(i, j)] = mat.AtUncheck(i, j);
                    }
                }
                Items = values;
            }
        }

        public MatrixSpanDebugView(Matrix mat)
            : this(mat.View) { }

        public long Rows { get; }
        public long Cols { get; }
        public long RowStride { get; }
        public long ColStride { get; }

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public readonly bool TooLong => Rows * Cols > TooLongLimit;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public const int TooLongLimit = int.MaxValue / 2;
    }
}
