using DDLA.BLAS;
using DDLA.Misc;
using DDLA.Misc.Pools;
using DDLA.Utilities;
using System.Runtime.CompilerServices;
using System.Text;

namespace DDLA.Core;

public partial class MArray
{
    #region Properties
    internal double[] Data { get; }

    internal int Offset { get; }

    internal int[] Metadata { get; }

    public ContinuousInfo Continuous { get; }

    public int Rank { get; }

    /// <summary>
    /// Indecate the state of the array.
    /// </summary>
    /// <remarks>
    /// It is a combination of <see cref="ArrayState"/> flags. So use <see cref="Enum.HasFlag(Enum)"/> to check the state.
    /// </remarks>
    public ArrayState State
    {
        get
        {
            ArrayState state = ArrayState.Default;
            ContinuousInfo info = Continuous;
            if (info.Layers[0].Stride == 1)
                state |= ArrayState.FortranStyle;

            if (info.Layers[^1].Stride != 1)
            {
                state |= ArrayState.Broken;
                state |= ArrayState.Segmented;
            }
            else
            {
                state |= ArrayState.CStyle;
                if (info.NumLayer != 1)
                {
                    state |= ArrayState.Segmented;
                }
            }
            return state;
        }
    }

    public ref double GetHeadRef()
    {
        if (Rank == 0)
            return ref Unsafe.NullRef<double>();
        return ref Data[Offset];
    }

    public ReadOnlySpan<int> Lengths => Metadata.AsSpan(0, Rank);

    public ReadOnlySpan<int> Strides => Metadata.AsSpan(Rank, Rank);

    public int Size => Lengths.Product();

    public bool IsScalar => Size == 1;
    #endregion

    #region Constructors
    public MArray(double[] data, ReadOnlySpan<int> shape)
    {
        Data = data;
        Offset = 0;
        Rank = shape.Length;
        var metadata = new int[Rank * 2];
        Metadata = metadata;
        if (shape.Length == 0)
        {
            Continuous = new(metadata);
            return;
        }
        int stride = 1;
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            int length = shape[i];
            metadata[i] = length;
            metadata[Rank + i] = stride;
            stride *= length;
        }
        ArgumentOutOfRangeException.ThrowIfGreaterThan(
            stride, data.Length, nameof(shape));
        Continuous = new(metadata);
    }

    public MArray(double[] data,
        int offset, ReadOnlySpan<int> shape)
    {
        Data = data;
        Offset = offset;
        Rank = shape.Length;
        var metadata = new int[Rank * 2];
        Metadata = metadata;
        if (shape.Length == 0)
        {
            Continuous = new(metadata);
            return;
        }
        int stride = 1;
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            int length = shape[i];
            metadata[i] = length;
            metadata[Rank + i] = stride;
            stride *= length;
        }
        ArgumentOutOfRangeException.ThrowIfGreaterThan(
            offset + stride, data.Length, nameof(shape));
        Continuous = new(metadata);
    }

    public MArray(double[] data,
        ReadOnlySpan<int> shape, ReadOnlySpan<int> strides)
    {
        Data = data;
        Offset = 0;
        Rank = shape.Length;
        var metadata = new int[Rank * 2];
        Metadata = metadata;
        if (shape.Length == 0)
        {
            Continuous = new(metadata);
            return;
        }
        int totalLength = 1;
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            int length = shape[i];
            int stride = strides[i];
            metadata[i] = length;
            metadata[Rank + i] = stride;
            totalLength += (length - 1) * stride;
        }
        ArgumentOutOfRangeException.ThrowIfGreaterThan(
            totalLength, data.Length, nameof(shape));
        Continuous = new(metadata);
    }

    public MArray(double[] data,
        int offset, ReadOnlySpan<int> shape, ReadOnlySpan<int> strides)
    {
        Data = data;
        Offset = offset;
        Rank = shape.Length;
        var metadata = new int[Rank * 2];
        Metadata = metadata;
        if (shape.Length == 0)
        {
            Continuous = new(metadata);
            return;
        }
        int totalLength = 1;
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            int length = shape[i];
            int stride = strides[i];
            metadata[i] = length;
            metadata[Rank + i] = stride;
            totalLength += (length - 1) * stride;
        }
        ArgumentOutOfRangeException.ThrowIfGreaterThan(
            offset + totalLength, data.Length, nameof(shape));
        Continuous = new(metadata);
    }

    public MArray(double[] data, int offset, int[] metadata, int rank)
    {
        Data = data;
        Offset = offset;
        Metadata = metadata;
        Rank = rank;
        Continuous = new(metadata);
    }
    #endregion Constructors

    #region Alloc
    public static MArray CreateUninitialized(ReadOnlySpan<int> shape)
    {
        int length = shape.Product();
        long totalSize = length * 8;
        var sizeB = totalSize % 1024;
        totalSize /= 1024;
        var sizeKB = totalSize % 1024;
        totalSize /= 1024;
        var sizeMB = totalSize % 1024;
        totalSize /= 1024;
        var sizeGB = totalSize;
        if (length * 8 > Array.MaxLength)
        {
            throw new ArgumentOutOfRangeException(nameof(shape),
                "The array is too large! It need to alloc " +
                $"{sizeGB} GB {sizeMB} MB {sizeKB} KB {sizeB} Byte.");
        }
        int size = length;
        try
        {
            double[] data = GC.AllocateUninitializedArray<double>(size);
            return new MArray(data, shape);
        }
        catch (Exception e)
        {
            throw new ArgumentOutOfRangeException(
                "The array is too large! It need to alloc " +
                $"{sizeGB} GB {sizeMB} MB {sizeKB} KB {sizeB} Byte.",
                innerException: e);
        }
    }

    public static MArray Create(params ReadOnlySpan<int> shape)
    {
        int length = shape.Product();
        double[] data = new double[length];
        return new MArray(data, shape);
    }

    public static MArray Create(ReadOnlySpan<int> shape, double val)
    {
        int length = shape.Product();
        double[] data = new double[length];
        data.AsSpan().Fill(val);
        return new MArray(data, shape);
    }

    public static MArray CreateUninitialized(ReadOnlySpan<int> shape,
        ReadOnlySpan<int> strides)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual
            (shape.Length, strides.Length);
        int rank = shape.Length;
        var metadata = new int[rank * 2];
        int totalLength = 1;
        for (int i = 0; i < shape.Length; i++)
        {
            int length = shape[i];
            int stride = strides[i];
            metadata[i] = length;
            metadata[rank + i] = stride;
            totalLength += (length - 1) * stride;
        }
        if (totalLength > Array.MaxLength)
            throw new ArgumentOutOfRangeException(nameof(shape),
                "The array is too large!");
        double[] data = GC.
            AllocateUninitializedArray<double>(totalLength);
        return new MArray(data, 0, metadata, rank);
    }

    public static MArray Create(ReadOnlySpan<int> shape,
        ReadOnlySpan<int> strides)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual
            (shape.Length, strides.Length);
        int rank = shape.Length;
        var metadata = new int[rank * 2];
        int totalLength = 1;
        for (int i = 0; i < shape.Length; i++)
        {
            int length = shape[i];
            int stride = strides[i];
            metadata[i] = length;
            metadata[rank + i] = stride;
            totalLength += (length - 1) * stride;
        }
        if (totalLength > Array.MaxLength)
            throw new ArgumentOutOfRangeException(nameof(shape),
                "The array is too large!");
        double[] data = new double[totalLength];
        return new MArray(data, 0, metadata, rank);
    }

    public static MArray Create(ReadOnlySpan<int> shape,
        ReadOnlySpan<int> strides, double val)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual
            (shape.Length, strides.Length);
        int rank = shape.Length;
        var metadata = new int[rank * 2];
        int totalLength = 1;
        for (int i = 0; i < shape.Length; i++)
        {
            int length = shape[i];
            int stride = strides[i];
            metadata[i] = length;
            metadata[rank + i] = stride;
            totalLength += (length - 1) * stride;
        }
        if (totalLength > Array.MaxLength)
            throw new ArgumentOutOfRangeException(nameof(shape),
                "The array is too large!");
        double[] data = new double[totalLength];
        data.AsSpan().Fill(val);
        return new MArray(data, 0, metadata, rank);
    }

    public MArray UninitializedLike()
        => CreateUninitialized(Lengths);

    public static MArray UninitializedLike(MArray array)
        => CreateUninitialized(array.Lengths);

    public MArray ZeroLike()
        => Create(Lengths);

    public static MArray ZeroLike(MArray array)
        => Create(array.Lengths);

    public MArray FillLike(double val)
        => Create(Lengths, val);

    public static MArray FillLike(MArray array, double val)
        => Create(array.Lengths, val);

    public MArray Clone()
    {
        var copied = UninitializedLike();
        BlasProvider.Copy(this, copied);
        return copied;
    }
    #endregion Alloc

    #region String

    public string MetaDataString()
    {
        using var _ = StringBuilderPool.Borrow(out var sb);
        MetaDataString(sb);
        return sb.ToString();
    }

    public void MetaDataString(StringBuilder sb)
    {
        sb
        .AppendLine($"Shape: ({string.Join(", ", Lengths.ToArray())})")
        .AppendLine($"Data Type: {typeof(double).Name}")
        .AppendLine($"Memory Usage: {Size * Unsafe.SizeOf<double>()} Byte")
        ;
    }

    public void Print(int start = 6, int end = 4,
        bool printMetadata = true, string? format = null)
    {
        using var _ = StringBuilderPool.Borrow(out var sb);
        ToString(sb, start, end, printMetadata, format);
        Console.WriteLine(sb.ToString());
    }

    public string ToString(int start = 6, int end = 4,
        bool printMetadata = true, string? format = null)
    {
        using var _ = StringBuilderPool.Borrow(out var sb);
        ToString(sb, start, end, printMetadata, format);
        return sb.ToString();
    }

    public void ToString(StringBuilder sb, int start = 6, int end = 4,
        bool printMetadata = true, string? format = null)
    {
        if (printMetadata)
            MetaDataString(sb);
        var segements = AsSegements();
        int dimLength = Math.Max(1, Rank);
        sb.Append('[', dimLength);
        if (segements.MoveNext())
        {
            segements.Current.ToString(sb, start, end, format);
            while (segements.MoveNext())
            {
                sb.Append(']', segements.Step)
                .Append(',')
                .AppendLine()
                .Append('[', segements.Step);
                segements.Current.ToString(sb, start, end, format);
            }
        }
        sb.Append(']', dimLength);
    }

    public MArraySegements AsSegements()
        => new(this);

    public MArrayEnumerator GetEnumerator()
        => new(this);
    #endregion

    #region Lowering

    public Vector AsVector(bool shouldCheck = false)
    {
        if (shouldCheck && Rank != 1)
            throw new Exception("The marray is not 1D!");
        return new(Data, Offset,
            Lengths[0], Strides[0]);
    }
    public Matrix AsMatrix(bool shouldCheck = false)
    {
        if (shouldCheck && Rank != 2)
            throw new Exception("The marray is not 2D!");
        return new Matrix(Data, Offset,
            Lengths[0], Lengths[1],
            Strides[0], Strides[1]);
    }
    #endregion Lowering

    public ref struct MArraySegements
    {
        private MArray Array { get; }
        public int Index { get; private set; }
        public int Length { get; }
        public int Batch { get; }
        public int Stride { get; }
        public readonly Span<int> StateSpan => state;

        public Span<int> state;

        public ReadOnlySpan<int> dimLengths;

        public ReadOnlySpan<int> dimStrides;

        public VectorView Current { get; set; }

        public int Step { get; set; }

        public MArraySegements(MArray array)
        {
            if (array.Rank == 0)
            {
                // For empty array, we can set it to the invalid state.
                Array = array;
                ReadOnlySpan<int> lengths = array.Lengths;
                Batch = lengths[0];
                Length = 0;
                Stride = 1;
                state = [];
                dimLengths = [];
                dimStrides = [];
                Index = 0;
                Current = VectorView.Empty;
            }
            else if (array.Rank == 1)
            {
                // For 1D array, we can use the VectorSpan directly.
                Array = array;
                ReadOnlySpan<int> lengths = array.Lengths;
                Batch = lengths[0];
                Length = 1;
                Stride = array.Strides[0];
                state = [];
                dimLengths = [];
                dimStrides = [];
                Index = -1;
            }
            else
            {
                Array = array;
                ReadOnlySpan<int> lengths = array.Lengths;
                Batch = lengths[^1];
                Length = lengths[..^1].Product();
                Stride = array.Strides[^1];
                state = new int[Array.Rank - 1];
                var rank = Array.Rank;
                dimLengths = Array.Metadata.AsSpan(0, rank - 1);
                dimStrides = Array.Metadata.AsSpan(rank, rank - 1);
                Index = -1;
            }
        }

        public bool MoveNext()
        {
            if (Length <= 1)
            {
                if (Index >= 0)
                    return false;
                else
                {
                    Index = 0;
                    Current = new(Array.Data, Array.Offset,
                        Batch, Stride);
                    return true;
                }
            }
            int dimLength = dimLengths.Length;
            var stateSpan = StateSpan;
            if (Index > -1)
            {
                Step = IntUtils.IncrementIndexLeft(dimLength - 1, stateSpan, dimLengths);
            }
            Index++;
            var stridesSpan = Array.Strides[..^1];
            int index = Array.Offset + IntUtils.Dot(stateSpan, stridesSpan);
            Current = new(Array.Data, index, Batch, Stride);
            return Index < Length;
        }
    }
    public ref struct MArrayEnumerator(MArray array)
    {
        MArraySegements sequences = new(array);

        int elementIndex = -1;

        public readonly ref double Current =>
            ref sequences.Current[elementIndex];

        public bool MoveNext()
        {
            elementIndex = (elementIndex + 1) % sequences.Batch;
            if (elementIndex == 0)
                return sequences.MoveNext();
            return true;
        }
    }
}
