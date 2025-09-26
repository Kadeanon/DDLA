using DDLA.BLAS;
using DDLA.Misc;
using DDLA.Misc.Flags;
using DDLA.UFuncs;
using DDLA.UFuncs.Operators;
using System.Collections;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace DDLA.Core;

/// <summary>
/// A span of a MatrixView with a reference to the first element.
/// </summary>
/// <typeparam name="double">The type of the elements in the MatrixView.</typeparam>
[DebuggerDisplay("{(long)Rows}x{(long)Cols}")]
[DebuggerTypeProxy(typeof(MatrixView.MatrixSpanDebugView))]
public class Matrix : IEnumerable<double>
{
    #region Properties
    internal double[] Data { get; }

    internal int Offset { get; set; }

    public int Rows { get; internal set; }

    public int Cols { get; internal set; }

    public int RowStride { get; internal set; }

    public int ColStride { get; internal set; }

    public int DiagOffset { get; }

    public int MinDim => Math.Min(Rows, Cols);

    public int MaxDim => Math.Max(Rows, Cols);

    public SingleIndice RowIndice => new SingleIndice(Rows, RowStride);

    public SingleIndice ColIndice => new SingleIndice(Cols, ColStride);

    public bool IsEmpty => Rows == 0 || Cols == 0;

    public int Size => Rows * Cols;

    public MatrixView View => new(Data, Offset,
        Rows, Cols, RowStride, ColStride, DiagOffset);
    #endregion Properties

    #region Constructors
    public Matrix(double[] array)
    {
        ArgumentNullException.ThrowIfNull(array, nameof(array));
        Data = array;
        Offset = 0;
        int length = array.Length;
        Rows = length;
        Cols = 1;
        RowStride = 1;
        ColStride = 1;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Matrix(double[] array, int offset)
    {
        ArgumentNullException.ThrowIfNull(array, nameof(array));
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual
            (offset, array.Length, nameof(offset));
        Data = array;
        Offset = offset;
        int length = array.Length - offset;
        Rows = length;
        Cols = 1;
        RowStride = 1;
        ColStride = 1;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Matrix(double[] array, int rows, int cols)
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
        RowStride = cols;
        ColStride = 1;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Matrix(double[] array, int offset, int rows, int cols)
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
        RowStride = cols;
        ColStride = 1;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Matrix(double[] array, int offset, int rows, int cols, int rowStride)
    {
        ArgumentNullException.ThrowIfNull(array, nameof(array));
        ArgumentOutOfRangeException.ThrowIfNegative(offset, nameof(offset));
        ArgumentOutOfRangeException.ThrowIfNegative(rows, nameof(rows));
        ArgumentOutOfRangeException.ThrowIfNegative(cols, nameof(cols));
        int length = (rows - 1) * rowStride + cols;
        ArgumentOutOfRangeException.ThrowIfLessThan(array.Length, offset + length,
            nameof(array));
        Data = array;
        Offset = offset;
        Rows = rows;
        Cols = cols;
        RowStride = rowStride;
        ColStride = 1;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Matrix(double[] array, int offset, int rows, int cols,
        int rowStride, int colStride, int diagOffset = 0)
    {
        ArgumentNullException.ThrowIfNull(array, nameof(array));
        ArgumentOutOfRangeException.ThrowIfNegative(offset, nameof(offset));
        ArgumentOutOfRangeException.ThrowIfNegative(rows, nameof(rows));
        ArgumentOutOfRangeException.ThrowIfNegative(cols, nameof(cols));
        int last = offset + (rows - 1) * rowStride + (cols - 1) * colStride + 1;
        ArgumentOutOfRangeException.ThrowIfLessThan(array.Length, last, nameof(array));
        Data = array;
        Offset = offset;
        Rows = rows;
        Cols = cols;
        RowStride = rowStride;
        ColStride = colStride;
        DiagOffset = diagOffset;
    }


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Matrix(MatrixView mat)
    {
        Data = mat.Data;
        Offset = mat.Offset;
        Rows = mat.Rows;
        Cols = mat.Cols;
        RowStride = mat.RowStride;
        ColStride = mat.ColStride;
        DiagOffset = mat.DiagOffset;
    }


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Matrix(VectorView vector)
    {
        Data = vector.Data;
        Offset = vector.Offset;
        Rows = vector.Length;
        RowStride = vector.Stride;
        Cols = 1;
        ColStride = Rows * vector.Stride;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Matrix(VectorView vector, int rows, int cols)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(vector.Length, rows * cols,
            nameof(vector));

        Data = vector.Data;
        Offset = vector.Offset;
        Rows = rows;
        RowStride = vector.Stride;
        Cols = cols;
        ColStride = rows * vector.Stride;
    }
    #endregion Constructors

    #region Alloc
    public static Matrix Create(int rows, int cols, bool uninited = false)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(rows, nameof(rows));
        ArgumentOutOfRangeException.ThrowIfNegative(cols, nameof(cols));
        int length = rows * cols;
        ArgumentOutOfRangeException.ThrowIfGreaterThan(length, Array.MaxLength, "size");
        double[] data = uninited ? GC.AllocateUninitializedArray<double>(length)
            : new double[length];
        return new Matrix(data, 0, rows, cols);
    }

    public static Matrix Filled(int rows, int cols, double val)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows, nameof(rows));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(cols, nameof(cols));
        int length = rows * cols;
        ArgumentOutOfRangeException.ThrowIfGreaterThan(length, Array.MaxLength, "size");
        double[] data = new double[length];
        Array.Fill(data, val);
        return new Matrix(data, 0, rows, cols, cols, 1);
    }

    public static Matrix Eyes(int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length, nameof(length));
        int size = length * length;
        double[] data = new double[size];
        var diagStride = length + 1;
        for (int i = 0; i < length; i++)
        {
            data[i * diagStride] = 1.0;
        }
        return new Matrix(data, 0, length, length, length, 1);
    }

    public static Matrix Diagonals(VectorView vector)
    {
        if (vector.Length == 0)
            return new(Array.Empty<double>());
        var result = Create(vector.Length, vector.Length);
        vector.CopyTo(result.Diag);
        return result;
    }

    public static Matrix RandomDense(int rows, int cols, Random? random = null)
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

    public static Matrix RandomTriangle(int dim,
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

    public static Matrix RandomSPD(int dim, Random? random = null)
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

    public static Matrix RandomSymmetric(int dim,
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

    public Matrix EmptyLike(bool clear = true)
        => Create(Rows, Cols, uninited: !clear);

    public Matrix FilledLike(double val)
        => Filled(Rows, Cols, val);
    #endregion Alloc

    #region Accessor and Slicer
    public ref double this[int row, int col]
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

    public ref double this[Index rowIndex, Index colIndex]
    {
        get
        {
            var row = rowIndex.GetOffset(Rows);
            if (row < 0 || row >= Rows)
                throw new ArgumentOutOfRangeException(nameof(rowIndex), $"Row index {row} is out of range.");
            var col = colIndex.GetOffset(Cols);
            if (col < 0 || col >= Cols)
                throw new ArgumentOutOfRangeException(nameof(colIndex), $"Column index {col} is out of range.");
            return ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(Data), Offset + row * RowStride + col * ColStride);
        }
    }

    public Matrix this[Range rows, Range cols]
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
            value.View.CopyTo(View.SliceSubUncheck(startRow, lengthRow, startCol, lengthCol));
        }
    }

    public Vector this[Index rowIndex, Range cols]
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
            value.View.CopyTo(View.GetRowUncheck(row));
        }
    }

    public Vector this[Range rows, Index colIndex]
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
            value.View.CopyTo(View.SliceColUncheck(col, startRow, lengthRow));
        }
    }

    public ref double GetHeadRef()
    {
        if (IsEmpty)
            return ref Unsafe.NullRef<double>();
        return ref Data[Offset];
    }

    public Vector GetRow(int row)
    {
        if (row < 0) row += Rows;
        if (row < 0 || row >= Rows)
            throw new ArgumentOutOfRangeException(nameof(row),
                $"Row index {row} is out of range.");
        return GetRowUncheck(row);
    }

    public Vector GetColumn(int col)
    {
        if (col < 0) col += Cols;
        if (col < 0 || col >= Cols)
            throw new ArgumentOutOfRangeException(nameof(col),
                $"Column index {col} is out of range.");
        return GetColUncheck(col);
    }

    public Vector Diag
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

    internal ref double AtUncheck(int row, int col)
        => ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(Data),
            Offset + row * RowStride + col * ColStride);

    internal Vector GetRowUncheck(int row)
        => new(Data, Offset + row * RowStride, Cols, ColStride);

    internal Vector SliceRowUncheck(int row, int colStart)
        => new(Data, Offset + row * RowStride + colStart * ColStride,
            Cols - colStart, ColStride);

    internal Vector SliceRowUncheck(int row, int colStart, int colLength)
        => new(Data, Offset + row * RowStride + colStart * ColStride,
            colLength, ColStride);

    internal Vector GetColUncheck(int col)
        => new(Data, Offset + col * ColStride, Rows, RowStride);

    internal Vector SliceColUncheck(int col, int rowStart)
        => new(Data, Offset + rowStart * RowStride + col * ColStride,
            Rows - rowStart, RowStride);

    internal Vector SliceColUncheck(int col, int rowStart, int rowLength)
        => new(Data, Offset + rowStart * RowStride + col * ColStride,
            rowLength, RowStride);

    internal Matrix SliceSubUncheck(int rowStart, int colStart)
        => new(Data, Offset + rowStart * RowStride + colStart * ColStride,
            Rows - rowStart, Cols - colStart, RowStride, ColStride);

    internal Matrix SliceSubUncheck(int rowStart, int rowLength, int colStart, int colLength)
        => new(Data, Offset + rowStart * RowStride + colStart * ColStride,
            rowLength, colLength, RowStride, ColStride);
    #endregion Accessor and Slicer

    #region Memory
    public void FlattenTo(Span<double> target)
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

    public void Fill(double val)
    {
        BlasProvider.Set(val, View);
    }

    public void Fill(double val, UpLo upLo, bool ignoreDiag = false)
    {
        BlasProvider.Set(ignoreDiag ? DiagType.Unit : DiagType.NonUnit, upLo, val, View);
    }

    public void Clear()
    {
        BlasProvider.Set(0.0, View);
    }

    public void Clear(UpLo upLo, bool ignoreDiag = false)
    {
        BlasProvider.Set(ignoreDiag ? DiagType.Unit: DiagType.NonUnit, upLo, 0.0, View);
    }

    public Matrix Transpose()
    {
        return new Matrix(Data, Offset,
                rows: Cols, cols: Rows,
                rowStride: ColStride, 
                colStride: RowStride);
    }

    public void CopyTo(Matrix other)
    {
        BlasProvider.Copy(this, other);
    }

    public Matrix Clone()
    {
        var target = Create(Rows, Cols, true);
        BlasProvider.Copy(this, target);
        return target;
    }

    public void MakeTr(UpLo uplo)
    {
        BlasProvider.MakeTr(View, uplo);
    }

    public void MakeSy(UpLo uplo)
    {
        BlasProvider.MakeSy(View, uplo);
    }

    public Vector Flatten(bool forceCopy = false)
    {
        if (!forceCopy && RowStride == Cols * ColStride)
        {
            return new Vector(Data, Offset, Size, ColStride);
        }
        double[] arr = new double[Size];
        FlattenTo(arr);
        return new(arr);
    }

    public Matrix Expand(int length = 1)
    {
        var result = Create(
            Rows + length, Cols + length, uninited: true);
        var slice =
            result[..Rows, ..Cols];
        CopyTo(slice);
        return result;
    }

    internal Span<double> GetSpan()
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
    public Matrix MakeRowMajor()
    {
        if (ColStride == 1)
        {
            return this; // Already continuous
        }
        return Clone();
    }
    #endregion Memory

    #region IEnumerable
    IEnumerator<double> IEnumerable<double>.GetEnumerator()
        => new MatrixView.MatrixEnumberator(this);

    IEnumerator IEnumerable.GetEnumerator()
        => new MatrixView.MatrixEnumberator(this);

    public MatrixView.RefMatrixEnumberator GetEnumerator()
        => new(this);
    #endregion IEnumerable

    #region Math
    public static Matrix operator +(Matrix self)
    {
        return self.Clone();
    }

    public static Matrix operator +(Matrix left, Matrix right)
    {
        right = right.Clone();
        BlasProvider.Add(left, right);
        return right;
    }

    public static Matrix operator +(double left, Matrix right)
        => right + left;

    public static Matrix operator +(Matrix left, double right)
    {
        var res = left.EmptyLike();
        UFunc.Map<AddOperator<double>, double>
            (left, right, res, default);
        return res;
    }

    public static Matrix operator -(Matrix self)
    {
        var res = self.EmptyLike();
        UFunc.Map<NegateOperator<double>>
            (self, res, default);
        return res;
    }

    public static Matrix operator -(Matrix left, Matrix right)
    {
        left = left.Clone();
        BlasProvider.Sub(right, left);
        return left;
    }

    public static Matrix operator -(Matrix left, double right)
    {
        var res = left.EmptyLike();
        UFunc.Map<SubtractOperator<double>, double>
            (left, right, res, default);
        return res;
    }

    public static Matrix operator -(double left, Matrix right)
    {
        var res = right.EmptyLike();
        UFunc.Map
            <SwappedOp<SubtractOperator<double>, double, double>, double>
            (right, left, res, default);
        return res;
    }

    public static Matrix operator *(double left, Matrix right)
    {
        var dest = Create(right.Rows, right.Cols, uninited: true);
        BlasProvider.Scal2(left, right, dest);
        return dest;
    }

    public static Matrix operator *(Matrix left, double right)
        => right * left;

    public static Matrix operator /(Matrix left, double right)
        => left * (1.0 / right);

    public static VectorView operator *(Matrix left, VectorView right)
    {
        var dest = VectorView.Create(left.Rows, true);
        BlasProvider.GeMV(1.0, left, right, 0.0, dest);
        return dest;
    }

    public static Matrix operator *(Matrix left, Matrix right)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(left.Cols, right.Rows,
            "Matrix multiplication requires the number of columns in the left MatrixView " +
            "to be equal to the number of rows in the right MatrixView.");
        int m = left.Rows;
        int n = right.Cols;
        var dest = Create(m, n, uninited: true);
        BlasProvider.GeMM(1.0, left, right, 0.0, dest);
        return dest;
    }

    public Matrix AddedBy(MatrixView other)
    {
        BlasProvider.Add(other, this);
        return this;
    }

    public Matrix SubtractedBy(MatrixView other)
    {
        BlasProvider.Sub(other, this);
        return this;
    }

    public Matrix ScaledBy(double alpha)
    {
        BlasProvider.Scal(alpha, this);
        return this;
    }

    public Matrix InvScaledBy(double alpha)
    {
        BlasProvider.InvScal(alpha, this);
        return this;
    }

    public Matrix ScaledTo(double alpha, MatrixView? output = null)
    {
        var result = output ?? EmptyLike();
        BlasProvider.Scal(alpha, result);
        return new(result);
    }

    public Matrix Multify(MatrixView other, MatrixView? output = null)
        => Multify(1.0, other, output);

    public Matrix Multify(MatrixView other, double beta, MatrixView output)
        => Multify(1.0, other, beta, output);

    public Matrix Multify(double alpha, MatrixView other, MatrixView? output = null)
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
            result = Create(m, n);
        }
        BlasProvider.GeMM(alpha, this, other, 0.0, result);
        return new(result);
    }

    public Matrix Multify(double alpha, MatrixView other, double beta, MatrixView output)
    {
        var m = Rows;
        var k = Cols;
        ArgumentOutOfRangeException.ThrowIfNotEqual(other.Rows, k, nameof(other));
        var n = other.Cols;
        ArgumentOutOfRangeException.ThrowIfNotEqual(output.Rows, m, nameof(output));
        ArgumentOutOfRangeException.ThrowIfNotEqual(output.Cols, n, nameof(output));
        BlasProvider.GeMM(alpha, this, other, beta, output);
        return new(output);
    }

    public Vector Multify(VectorView other, VectorView? output = null)
        => Multify(1.0, other, output);

    public Vector Multify(VectorView other, double beta, VectorView output)
        => Multify(1.0, other, beta, output);

    public Vector Multify(double alpha, VectorView other, VectorView? output = null)
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
        return new(result);
    }

    public Vector Multify(double alpha, VectorView other, double beta, VectorView output)
    {
        var m = Rows;
        var k = Cols;
        ArgumentOutOfRangeException.ThrowIfNotEqual(other.Length, k, nameof(other));
        ArgumentOutOfRangeException.ThrowIfNotEqual(output.Length, m, nameof(output));
        BlasProvider.GeMV(alpha, this, other, beta, output);
        return new(output);
    }

    public void Rank1(VectorView x, VectorView y)
        => BlasProvider.GeR(1.0, x, y, this);

    public void Rank1(double alpha, VectorView x, VectorView y)
        => BlasProvider.GeR(alpha, x, y, this);

    public void Rank1(UpLo uplo, VectorView x, VectorView y)
    {
        if (uplo is UpLo.Dense)
            BlasProvider.GeR(1.0, x, y, this);
        else if (uplo is UpLo.Upper or UpLo.Lower)
            BlasProvider.SyR(uplo, 1.0, x, this);
        else
            throw new ArgumentException($"Matrix c must be upper or lower triangular!");
    }

    public void Rank1(UpLo uplo, double alpha, VectorView x, VectorView y)
    {
        if (uplo is UpLo.Dense)
            BlasProvider.GeR(alpha, x, y, this);
        else if (uplo is UpLo.Upper or UpLo.Lower)
            BlasProvider.SyR(uplo, alpha, x, this);
        else
            throw new ArgumentException($"Matrix c must be upper or lower triangular!");
    }

    public void ShiftDiag(double alpha)
        => BlasProvider.ShiftDiag(alpha, View);

    public void SwapCol(int i, int j)
        => BlasProvider.Swap(View.GetColumn(i), View.GetColumn(j));

    public void SwapRow(int i, int j)
        => BlasProvider.Swap(View.GetRow(i), View.GetRow(j));
    #endregion Math

    #region Ufunc
    public Matrix Pointwise<TUnaryAction>()
        where TUnaryAction : struct, IUnaryOperator<double, double>
    {
        var dest = EmptyLike();
        UFunc.Map<TUnaryAction>(this, dest, new());
        return dest;
    }

    public Matrix PointwiseAbs()
        => Pointwise<AbsOperator<double>>();
    public double Nrm1() =>
        BlasProvider.Nrm1(this);

    public double NrmF() =>
        BlasProvider.NrmF(this);

    public double NrmInf() =>
        BlasProvider.NrmInf(this);

    #endregion Ufunc

    public static implicit operator MatrixView(Matrix mat)
        => mat.View;

    public override string ToString() => View.ToString();
}
