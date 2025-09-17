using DDLA.BLAS;
using DDLA.Misc;
using DDLA.Misc.Pools;
using DDLA.UFuncs;
using DDLA.UFuncs.Operators;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace DDLA.Core;


[DebuggerDisplay("{Length}x1")]
[DebuggerTypeProxy(typeof(VectorDebugView))]
public readonly struct VectorView
{
    public double[] Data { get; }
    public int Offset { get; }
    public int Length { get; }
    public int Stride { get; }

    public readonly SingleIndice Indice => new(Length, Stride);


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public VectorView(double[] array)
    {
        ArgumentNullException.ThrowIfNull(array, nameof(array));
        Data = array;
        Offset = 0;
        Length = array.Length;
        Stride = 1;
    }


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public VectorView(double[] array, int start, int length, int step = 1)
    {
        ArgumentNullException.ThrowIfNull(array, nameof(array));
        ArgumentOutOfRangeException.ThrowIfNegative(start, nameof(start));
        ArgumentOutOfRangeException.ThrowIfLessThan(step, 1, nameof(step));
        ArgumentOutOfRangeException.ThrowIfNegative(length, nameof(length));
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(array.Length, start + (length - 1) * step);

        Data = array;
        Offset = start;
        Length = length;
        Stride = step;
    }

    public static VectorView Create(int length, bool uninited = false)
    {
        ArgumentOutOfRangeException.ThrowIfGreaterThan(length, Array.MaxLength, nameof(length));
        double[] data = uninited ? GC.AllocateUninitializedArray<double>(length)
            : new double[length];
        return new(data);
    }

    public static VectorView Create(int length, double val)
    {
        ArgumentOutOfRangeException.ThrowIfGreaterThan(length, Array.MaxLength, nameof(length));
        double[] data = GC.AllocateUninitializedArray<double>(length);
        data.AsSpan().Fill(val);
        return new(data);
    }

    public ref double this[int index] => ref At(index);

    public VectorView this[Range range]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            (var start, var length) = range.GetOffsetAndLength(Length);
            return Slice(start, length);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        set
        {
            (var start, var length) = range.GetOffsetAndLength(Length);
            value.CopyTo(Slice(start, length));
        }
    }

    public readonly bool IsEmpty => Length == 0;

    public override readonly bool Equals(object? obj) =>
            throw new NotSupportedException();

    public override readonly int GetHashCode() =>
        throw new NotSupportedException();

    public static VectorView Empty => default;

    public readonly VectorEnumerator GetEnumerator() => new(this);

    public VectorView MakeContinous(bool forceCopy = false)
    {
        if (!forceCopy && Stride == 1)
            return this;
        return Clone();
    }

    public readonly VectorView Clone()
    {
        if (IsEmpty)
            return new VectorView([]);
        var target = new VectorView(new double[Length]);
        BlasProvider.Copy(this, target);
        return target;
    }

    internal readonly ref double GetHeadRef()
    {
        if (IsEmpty)
            return ref Unsafe.NullRef<double>();
        return ref Data[Offset];
    }

    public readonly ref double GetPinnableReference()
        => ref GetHeadRef();

    internal readonly Span<double> GetSpan()
    {
        if (IsEmpty)
            throw new InvalidOperationException("Vector is empty.");
        return Data.AsSpan(Offset);
    }

    public ref struct VectorEnumerator
    {
        private readonly VectorView _span;

        private int _index;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal VectorEnumerator(VectorView span)
        {
            _span = span;
            _index = -1;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool MoveNext()
        {
            var index = _index + 1;
            if (index < _span.Length)
            {
                _index = index;
                return true;
            }

            return false;
        }

        public readonly ref double Current
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref _span[_index];
        }
    }

    public readonly unsafe void Clear()
    {
        if (Length == 0)
        {
            return;
        }
        ref double current = ref GetHeadRef();
        if (Stride == 1)
        {
            for (int i = 0; i < Length; i += int.MaxValue / 2)
            {
                int length = Math.Min(int.MaxValue / 2, Length - i);
                var span = MemoryMarshal.CreateSpan(ref current, length);
                span.Clear();
                current = ref Unsafe.Add(ref current, length);
            }
        }
        else
        {
            for (int i = 0; i < Length; i++)
            {
                current = default;
                current = ref Unsafe.Add(ref current, Stride);
            }
        }
        return;
    }

    public readonly unsafe void Fill(double value)
    {
        if (Length == 0)
            return;
        if (value == default)
        {
            Clear();
            return;
        }

        ref double current = ref Data[Offset];
        if (Stride == 1)
        {
            for (int i = 0; i < Length; i += int.MaxValue / 2)
            {
                int length = Math.Min(int.MaxValue / 2, Length - i);
                var span = MemoryMarshal.CreateSpan(ref current, length);
                span.Fill(value);
                current = ref Unsafe.Add(ref current, length);
            }
        }
        else
        {
            for (int i = 0; i < Length; i++)
            {
                current = value;
                current = ref Unsafe.Add(ref current, Stride);
            }
        }
        return;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly VectorView Slice(int start)
    {
        if (start > Length)
            throw new ArgumentOutOfRangeException(nameof(start), start, "Error: start index should be less than length.");

        return new VectorView(Data, Offset + start * Stride, Length - start, Stride);
    }

    public readonly void CopyTo(VectorView other)
    {
        ArgumentOutOfRangeException.
            ThrowIfLessThan(other.Length, Length, nameof(other));
        ref double current = ref GetHeadRef();
        ref double otherCurrent = ref other.GetHeadRef();
        for (int i = 0; i < Length; i++)
        {
            otherCurrent = current;
            current = ref Unsafe.Add(ref current, Stride);
            otherCurrent = ref Unsafe.Add(ref otherCurrent, other.Stride);
        }
    }

    public readonly ref double At(int index)
    {
        if (index < 0 || index >= Length)
            throw new IndexOutOfRangeException(nameof(index));
        return ref AtUncheck(index);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal readonly ref double AtUncheck(int index) => ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(Data), Offset + index * Stride);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly VectorView Slice(int start, int length)
    {
        if (start > Length)
            throw new ArgumentOutOfRangeException(nameof(start), start, "Error: start index should be less than length.");
        if (length < 0 || start + length > Length)
            throw new ArgumentOutOfRangeException(nameof(length), length, "Error: length should be greater than 0 and end of span should be in range.");
        return SliceUncheck(start, length);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal readonly VectorView SliceUncheck(int start) => new VectorView(Data, Offset + start * Stride, Length - start, Stride);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal readonly VectorView SliceUncheck(int start, int length) => new VectorView(Data, Offset + start * Stride, length, Stride);

    public readonly void FlattenTo(Span<double> target)
    {
        if (target.Length < Length)
            throw new ArgumentException("Error: target length should be greater than source length.", nameof(target));
        ref double current = ref Data[Offset];
        int length = Length;
        if (Stride == 1)
        {
            MemoryMarshal.CreateSpan(ref current, length).CopyTo(target);
            return;
        }
        ref double targetCurrent = ref MemoryMarshal.GetReference(target);
        for (int i = 0; i < length; i++)
        {
            targetCurrent = current;
            current = ref Unsafe.Add(ref current, Stride);
            targetCurrent = ref Unsafe.Add(ref targetCurrent, 1);
        }
    }

    public static double operator *(VectorView left, VectorView right)
        => BlasProvider.Dot(left, right);

    public static VectorView operator +(VectorView left, VectorView right)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(left.Length, right.Length);
        var result = left.Clone();
        BlasProvider.Add(right, result);
        return result;
    }

    public static VectorView operator -(VectorView left, VectorView right)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(left.Length, right.Length);
        var result = left.Clone();
        BlasProvider.Sub(right, result);
        return result;
    }

    public static VectorView operator *(double scalar, VectorView vector)
    {
        var result = vector.Clone();
        BlasProvider.Scal(scalar, result);
        return result;
    }

    public static VectorView operator *(VectorView vector, double scalar)
    {
        var result = vector.Clone();
        BlasProvider.Scal(scalar, result);
        return result;
    }

    public static VectorView operator /(VectorView vector, double scalar)
    {
        if (scalar == 0)
            throw new DivideByZeroException("Error: Division by zero.");
        var result = vector.Clone();
        BlasProvider.InvScal(scalar, result);
        return result;
    }

    public static VectorView operator -(VectorView vector)
    {
        var result = vector.Clone();
        BlasProvider.Scal(-1.0, result);
        return result;
    }

    public readonly void Added(VectorView other) => 
        BlasProvider.Add(other, this);

    public readonly void Added(double alpha, VectorView other) =>
        BlasProvider.Axpy(alpha, other, this);

    public readonly void Added(VectorView other, double beta) =>
        BlasProvider.Xpby(other, beta, this);

    public readonly void Added(double alpha, VectorView other, double beta) =>
        BlasProvider.Axpby(alpha, other, beta, this);

    public readonly void Subtracted(VectorView other) =>
        BlasProvider.Sub(other, this);

    public readonly double Dot(VectorView other) =>
        BlasProvider.Dot(this, other);

    public readonly Matrix Outer(VectorView other)
    {
        Matrix result = Matrix.Create(Length, other.Length);
        BlasProvider.GeR(1, this, other, result);
        return result;
    }

    public readonly Matrix T => new(Data, Offset, 1, Length,
        Length * Stride, Stride);

    public readonly VectorView LeftMul(MatrixView right, VectorView? output = null)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(right.Rows, Length, nameof(right));
        var result = output ?? Create(right.Cols, uninited:true);
        BlasProvider.GeMV(Misc.Flags.TransType.OnlyTrans,
            1.0, right, this, 0.0, result);
        return result;
    }

    public readonly VectorView LeftMul(MatrixView right, double beta, VectorView output)
    {
        BlasProvider.GeMV(Misc.Flags.TransType.OnlyTrans,
            1.0, right, this, beta, output);
        return output;
    }

    public readonly VectorView LeftMul(double alpha, MatrixView right, VectorView? output = null)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(right.Cols, Length, nameof(right));
        var result = output ?? Create(right.Rows, uninited: true);
        BlasProvider.GeMV(Misc.Flags.TransType.OnlyTrans,
            alpha, right, this, 0.0, result);
        return result;
    }

    public readonly VectorView LeftMul(double alpha, MatrixView right, double beta, VectorView output)
    {
        BlasProvider.GeMV(Misc.Flags.TransType.OnlyTrans,
            alpha, right, this, beta, output);
        return output;
    }

    public double Nrm1() =>
        BlasProvider.Nrm1(this);

    public double NrmF() =>
        BlasProvider.NrmF(this);

    public double NrmInf() =>
        BlasProvider.NrmInf(this);

    public readonly double Max() =>
        BlasProvider.NrmInf(this);

    public readonly double SumSq() 
        => UFunc.Sum<SquareOperator<double>> (this);

    public readonly double Sum() 
        => UFunc.Sum<IdentityOperator<double>> (this);

    public readonly double SumAbs() 
        => UFunc.Sum<AbsOperator<double>> (this);

    public VectorView Scale(double scalar, VectorView? output = null)
    {
        var result = output ?? Create(Length, uninited: true);
        BlasProvider.Scal2(scalar, this, result);
        return result;
    }

    public VectorView Scaled(double scalar)
    {
        if (IsEmpty)
            return this;
        BlasProvider.Scal(scalar, this);
        return this;
    }

    public VectorView InvScaled(double scalar)
    {
        if (IsEmpty)
            return this;
        BlasProvider.Scal(1 / scalar, this);
        return this;
    }

    public VectorView Normalize()
    {
        if (IsEmpty)
            return this;
        var norm = NrmF();
        if (norm == 0)
            throw new DivideByZeroException("Error: Cannot normalize a zero vector.");
        var result = Clone();
        BlasProvider.Scal(1.0 / norm, result);
        return result;
    }

    public readonly VectorView Normalized()
    {
        if (IsEmpty)
            return this;
        BlasProvider.InvScal(NrmF(), this);
        return this;
    }

    public override string ToString()
    {
        using var handle = StringBuilderPool.Borrow(out var sb);
        ToString(sb, 6, 4, null);
        return $"VectorView[{Length}]{sb}";
    }

    /// <summary>
    /// Returns a string representation of the elements in the span.
    /// </summary>
    /// <param name="sb"> The <see cref="StringBuilder"/> object to write the string</param>
    /// <param name="start">If the span is too long, determines how many element to write before '...'.</param>
    /// <param name="end">If the span is too long, determines how many element to write after '...'.</param>
    public void ToString(StringBuilder sb, int start = 6, int end = 4, string? format = null)
    {
        if (Length == 0)
        {
            return;
        }
        ref double current = ref GetHeadRef();

        sb.Append('[');
        if (format is not null)
        {
            if (Length <= start + end)
            {
                sb.Append(current.ToString(format));
                int i = 1;
                for (; i < Length; i++)
                {
                    current = ref Unsafe.Add(ref current, Stride);
                    sb.Append(", ");
                    sb.Append(current.ToString(format));
                }
            }
            else
            {
                for (int i = 0; i < start; i++)
                {
                    if (i > 0) sb.Append(", ");
                    sb.Append(this[i].ToString(format));
                }
                sb.Append(", ... ");
                for (int i = Length - end; i < Length; i++)
                {
                    if (i > 0) sb.Append(", ");
                    sb.Append(this[i].ToString(format));
                }
            }
        }
        else
        {
            if (Length <= start + end)
            {
                sb.Append(current);
                int i = 1;
                for (; i < Length; i++)
                {
                    current = ref Unsafe.Add(ref current, Stride);
                    sb.Append(", ");
                    sb.Append(current);
                }
            }
            else
            {
                for (int i = 0; i < start; i++)
                {
                    if (i > 0) sb.Append(", ");
                    sb.Append(this[i]);
                }
                sb.Append(", ... ");
                for (int i = Length - end; i < Length; i++)
                {
                    if (i > 0) sb.Append(", ");
                    sb.Append(this[i]);
                }
            }
        }
        sb.Append(']');
    }

}

public readonly ref struct VectorDebugView
{
    public readonly object Items;

    public VectorDebugView(Vector vector)
    {
        Length = vector.Length;
        Stride = vector.Stride;

        if (vector.IsEmpty || TooLong)
        {
            Items = Array.Empty<double>();
        }
        else
        {
            var data = new double[vector.Length];
            vector.FlattenTo(data);
            Items = data;
        }
    }

    public VectorDebugView(VectorView view)
    {
        Length = view.Length;
        Stride = view.Stride;

        if (view.IsEmpty || TooLong)
        {
            Items = Array.Empty<double>();
        }
        else
        {
            var data = new double[Length];
            VectorView self = new(data);
            view.CopyTo(self);
            Items = data;
        }
    }

    public long Length { get; }
    public long Stride { get; }

    [DebuggerBrowsable(DebuggerBrowsableState.Never)]
    public bool TooLong => Length > TooLongLimit;

    [DebuggerBrowsable(DebuggerBrowsableState.Never)]
    public const int TooLongLimit = int.MaxValue / 2;
}
