using DDLA.BLAS;
using DDLA.Misc;
using DDLA.Misc.Pools;
using DDLA.UFuncs;
using DDLA.UFuncs.Operators;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace DDLA.Core;

using BlasProvider = DDLA.BLAS.Managed.BlasProvider;

[DebuggerDisplay("{(long)Length}x1")]
[DebuggerTypeProxy(typeof(VectorDebugView))]
public class Vector
{
    public double[] Data { get; }
    public int Offset { get; }
    public int Length { get; }
    public int Stride { get; }

    public SingleIndice Indice => new(Length, Stride);


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector(double[] array)
    {
        ArgumentNullException.ThrowIfNull(array, nameof(array));
        Data = array;
        Offset = 0;
        Length = array.Length;
        Stride = 1;
    }


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector(double[] array, int start, int length, int step = 1)
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


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector(VectorView view)
    {
        ArgumentNullException.ThrowIfNull(view.Data, nameof(view));

        Data = view.Data;
        Offset = view.Offset;
        Length = view.Length;
        Stride = view.Stride;
    }

    public static Vector Create(int length, bool uninited = false)
    {
        ArgumentOutOfRangeException.ThrowIfGreaterThan(length, Array.MaxLength, nameof(length));
        double[] data = uninited ? GC.AllocateUninitializedArray<double>(length)
            : new double[length];
        return new(data);
    }

    public static Vector Create(int length, double val)
    {
        ArgumentOutOfRangeException.ThrowIfGreaterThan(length, Array.MaxLength, nameof(length));
        double[] data = GC.AllocateUninitializedArray<double>(length);
        data.AsSpan().Fill(val);
        return new(data);
    }

    public static Vector Random(int length, Random? random = null)
    {
        random ??= System.Random.Shared;

        var data = GC.AllocateUninitializedArray<double>(length);
        for (int i = 0; i < length; i++)
            data[i] = random.NextDouble();
        return new(data);
    }

    public ref double this[int index] => ref At(index);

    public Vector this[Range range]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            var(start, length) = range.GetOffsetAndLength(Length);
            return Slice(start, length);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        set
        {
            (var start, var length) = range.GetOffsetAndLength(Length);
            value.CopyTo(Slice(start, length));
        }
    }

    public bool IsEmpty => Length == 0;

    public override bool Equals(object? obj) =>
            throw new NotSupportedException();

    public override int GetHashCode() =>
        throw new NotSupportedException();

    public static Span<double> Empty => default;

    public VectorEnumerator GetEnumerator() => new(this);

    public Vector MakeContinous(bool forceCopy = false)
    {
        if (!forceCopy && Stride == 1)
            return this;
        return Clone();
    }

    public Vector Clone()
    {
        if (IsEmpty)
            return new Vector([]);
        var target = new Vector(new double[Length]);
        BlasProvider.Copy(this, target);
        return target;
    }

    internal ref double GetHeadRef()
    {
        if (IsEmpty)
            return ref Unsafe.NullRef<double>();
        return ref Data[Offset];
    }

    internal Span<double> GetSpan()
    {
        if (IsEmpty)
            throw new InvalidOperationException("Vector is empty.");
        return Data.AsSpan(Offset);
    }

    public ref struct VectorEnumerator
    {
        private readonly Vector _span;

        private int _index;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal VectorEnumerator(Vector span)
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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public unsafe void Clear()
    {
        if (Length == 0)
        {
            return;
        }
        ref double current = ref Data[Offset];
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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public unsafe void Fill(double value)
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
    public Vector Slice(int start)
    {
        if (start > Length)
            throw new ArgumentOutOfRangeException(nameof(start), start, "Error: start index should be less than length.");

        return new Vector(Data, Offset + start * Stride, Length - start, Stride);
    }

    public void CopyTo(VectorView other)
    {
        BlasProvider.Copy(this, other);
    }

    public void CopyTo(Span<double> target)
        => View.CopyTo(target);

    public void CopyFrom(VectorView other)
    {
        BlasProvider.Copy(this, other);
    }

    public void CopyFrom(ReadOnlySpan<double> source)
        => View.CopyFrom(source);

    public ref double At(int index)
    {
        if (index < 0 || index >= Length)
            throw new IndexOutOfRangeException(nameof(index));
        return ref AtUncheck(index);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal ref double AtUncheck(int index)
    {
        return ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(Data), Offset + index * Stride);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector Slice(int start, int length)
    {
        if (start > Length)
            throw new ArgumentOutOfRangeException(nameof(start), start, "Error: start index should be less than length.");
        if (length < 0 || start + length > Length)
            throw new ArgumentOutOfRangeException(nameof(length), length, "Error: length should be greater than 0 and end of span should be in range.");
        return SliceUncheck(start, length);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal Vector SliceUncheck(int start, int length)
    {
        return new Vector(Data, Offset + start * Stride, length, Stride);
    }

    public void FlattenTo(Span<double> target)
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

    public static double operator *(Vector left, Vector right)
        => BlasProvider.Dot(left, right);

    public static Vector operator +(Vector left, Vector right)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(left.Length, right.Length);
        var result = left.Clone();
        BlasProvider.Add(right, result);
        return result;
    }

    public static Vector operator -(Vector left, Vector right)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(left.Length, right.Length);
        var result = left.Clone();
        BlasProvider.Sub(right, result);
        return result;
    }

    public static Vector operator *(double scalar, Vector vector)
    {
        ArgumentNullException.ThrowIfNull(vector, nameof(vector));
        var result = vector.Clone();
        BlasProvider.Scal(scalar, result);
        return result;
    }

    public static Vector operator *(Vector vector, double scalar)
    {
        ArgumentNullException.ThrowIfNull(vector, nameof(vector));
        var result = vector.Clone();
        BlasProvider.Scal(scalar, result);
        return result;
    }

    public static Vector operator /(Vector vector, double scalar)
    {
        ArgumentNullException.ThrowIfNull(vector, nameof(vector));
        if (scalar == 0)
            throw new DivideByZeroException("Error: Division by zero.");
        var result = vector.Clone();
        BlasProvider.InvScal(scalar, result);
        return result;
    }

    public static Vector operator -(Vector vector)
    {
        ArgumentNullException.ThrowIfNull(vector, nameof(vector));
        var result = vector.Clone();
        BlasProvider.Scal(-1.0, result);
        return result;
    }

    public void AddedBy(VectorView other) =>
        BlasProvider.Add(other, View);

    public void AddedBy(double alpha, VectorView other) =>
        BlasProvider.Axpy(alpha, other, View);

    public void AddedBy(VectorView other, double beta) =>
        BlasProvider.Xpby(other, beta, View);

    public void AddedBy(double alpha, VectorView other, double beta) =>
        BlasProvider.Axpby(alpha, other, beta, View);

    public void SubtractedBy(VectorView other) =>
        BlasProvider.Sub(other, View);

    public double Dot(Vector other) =>
        BlasProvider.Dot(View, other);

    public Matrix Outer(Vector other, Matrix? output = null)
    {
        Matrix result = output ?? Matrix.Create(Length, other.Length);
        BlasProvider.GeR(1, View, other, result);
        return result;
    }

    public Matrix T => new(Data, Offset, 1, Length,
        Length * Stride, Stride);

    public Vector LeftMul(MatrixView right, Vector? output = null)
    {
        var result = output ?? Create(right.Rows, uninited: true);
        BlasProvider.GeMV(Misc.Flags.TransType.OnlyTrans,
            1.0, right, View, 0.0, result);
        return result;
    }

    public Vector LeftMul(MatrixView right, double beta, Vector output)
    {
        BlasProvider.GeMV(Misc.Flags.TransType.OnlyTrans,
            1.0, right, View, beta, output);
        return output;
    }

    public Vector LeftMul(double alpha, MatrixView right, Vector? output = null)
    {
        var result = output ?? Create(right.Rows, uninited: true);
        BlasProvider.GeMV(Misc.Flags.TransType.OnlyTrans,
            alpha, right, View, 0.0, result);
        return result;
    }

    public Vector LeftMul(double alpha, MatrixView right, double beta, Vector output)
    {
        BlasProvider.GeMV(Misc.Flags.TransType.OnlyTrans,
            alpha, right, View, beta, output);
        return output;
    }

    public double Nrm1() =>
        BlasProvider.Nrm1(View);

    public double NrmF() =>
        BlasProvider.NrmF(View);

    public double NrmInf() =>
        BlasProvider.NrmInf(View);

    public double Max() =>
        UFunc.Reduce<MaxAggregationOperator<double>>(View);

    public double SumSq()
        => UFunc.Sum<SquareOperator<double>>(View);

    public double Sum()
        => UFunc.Sum<IdentityOperator<double>>(View);

    public double SumAbs()
        => UFunc.Sum<AbsOperator<double>>(View);

    public Vector Scale(double scalar, Vector? output = null)
    {
        var result = output ?? Create(Length, uninited: true);
        BlasProvider.Scal2(scalar, View, result.View);
        return result;
    }

    public Vector Scaled(double scalar)
    {
        if (IsEmpty)
            return this;
        BlasProvider.Scal(scalar, View);
        return this;
    }

    public Vector InvScaled(double scalar)
    {
        if (IsEmpty)
            return this;
        BlasProvider.InvScal(scalar, View);
        return this;
    }

    public Vector Normalize()
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

    public Vector Normalized()
    {
        if (IsEmpty)
            return this;
        BlasProvider.InvScal(NrmF(), this);
        return this;
    }

    public VectorView View
        => new(Data, Offset, Length, Stride);

    public static implicit operator VectorView
        (Vector vector) => vector.View;

    public override string ToString()
    {
        return ToString(4, 6, null);
    }

    public string ToString(int start = 4, int end = 6, string? format = null)
    {
        using var handle = StringBuilderPool.Borrow(out var sb);
        View.ToString(sb,  start, end, format);
        return sb.ToString();
    }
}
