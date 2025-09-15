using DDLA.BLAS;
using DDLA.Misc;
using DDLA.Utilities;
using System.Diagnostics.CodeAnalysis;

namespace DDLA.Core;

public partial class MArray
{
    public MArray Flatten(bool forceCopy = false)
    {
        if (Rank == 1 && !forceCopy)
        {
            return this;
        }
        int[] newShape = { Size };
        return Reshape(newShape, forceCopy);
    }

    public void FlattenTo(Span<double> destination)
    {
        if (destination.Length < Size)
        {
            throw new ArgumentException(
                "The destination span length must be " +
                "greater than or equal to the size of the array.",
                nameof(destination));
        }
        //TODO: more efficient way to flatten
        int i = 0;
        foreach (var item in this)
        {
            destination[i++] = item;
        }
    }

    public MArray Reshape(params ReadOnlySpan<int> shape) => Reshape(shape, true, false);

    public MArray Reshape(ReadOnlySpan<int> shape, bool allowCopy = true, bool forceCopy = false)
    {
        Span<int> tempShape = stackalloc int[shape.Length];
        shape.CopyTo(tempShape);
        //如果形状相同，直接返回
        if (Reshape_CheckShape(tempShape) && !forceCopy)
        {
            return this;
        }

        var view = Reshape_TryBuildHead(tempShape, out var head);
        if (!view || forceCopy)
        {
            if (!allowCopy)
            {
                throw new InvalidOperationException("Cannot reshape without copying");
            }
            return Reshape_Copy(tempShape);
        }
        else
        {
            return new(Data, Offset, head, tempShape.Length);
        }
    }

    private bool Reshape_CheckShape(Span<int> shape)
    {
        if (shape.Length == 0)
        {
            throw new ArgumentException("Shape cannot be empty");
        }
        int size = 1;
        int wildcard = -1;
        for (int i = 0; i < shape.Length; i++)
        {
            int shapeValue = shape[i];
            if (shapeValue <= 0)
            {
                if (shapeValue != -1)
                {
                    throw new ArgumentException("Shape values must be positive or -1 as wildcard");
                }
                if (wildcard != -1)
                {
                    throw new ArgumentException("Shape cannot contain more than one wildcard");
                }
                wildcard = i;
                continue;
            }
            size *= shape[i];
        }
        if (wildcard != -1)
        {
            (var quotient, var remainder) = Math.DivRem(Size, size);
            if (remainder != 0)
            {
                throw new ArgumentException("Cannot determine wildcard value, the size of array is not a multiple of the new size with other dims.");
            }
            shape[wildcard] = quotient;
        }
        else
        {
            if (size != Size)
            {
                throw new ArgumentException("Size of new shape must be equal to the size of the original shape");
            }
        }
        return shape.Length == Lengths.Length && shape.SequenceEqual(Lengths);
    }

    private bool Reshape_TryBuildHead(ReadOnlySpan<int> shape, out int[] head)
    {
        var rank = shape.Length;
        head = new int[rank * 2];
        ContinuousInfo info = new(Metadata, false);
        int newNumDims = shape.Length;
        Span<int> strides = stackalloc int[newNumDims];
        //Try check with continuous info
        var continuouslayers = info.Layers.AsSpan();
        Span<int> layers = stackalloc int[continuouslayers.Length];
        int numLayers = 0;
        for (var iLayer = 0; iLayer < continuouslayers.Length; iLayer++)
        {
            ref var layer = ref continuouslayers[iLayer];
            if (layer.IsHead)
            {
                layers[numLayers] = iLayer;
                numLayers++;
            }
        }
        layers = layers[..numLayers];
        numLayers--;
        ref ContinuousLayer layerHeader = ref continuouslayers[layers[numLayers]];
        int stride = layerHeader.Stride;
        int size = 1;
        for (int i = newNumDims - 1; i >= 0; i--)
        {
            strides[i] = stride;
            stride *= shape[i];
            size *= shape[i];
            if (size > layerHeader.BlockSize)
            {
                return false;
            }
            else if (size == layerHeader.BlockSize)
            {
                if (numLayers == 0)
                {
                    break;
                }
                numLayers--;
                layerHeader = ref continuouslayers[layers[numLayers]];
                stride = layerHeader.Stride;
                size = 1;
            }
        }
        for (int i = 0; i < newNumDims; i++)
        {
            head[i] = shape[i];
            head[i + rank] = strides[i];
        }
        return true;
    }

    private MArray Reshape_Copy(ReadOnlySpan<int> shape)
    {
        double[] arrayData = new double[shape.Product()];
        FlattenTo(arrayData);
        return new(arrayData, shape);
    }

    public bool TryReshapeWithoutCopy(ReadOnlySpan<int> shape,
        [NotNullWhen(true)] out MArray? result)
    {
        Span<int> tempShape = stackalloc int[shape.Length];
        shape.CopyTo(tempShape);
        if (Reshape_CheckShape(tempShape))
        {
            result = this;
            return true;
        }

        if (Reshape_TryBuildHead(tempShape, out var head))
        {
            result = new(Data, Offset, head, tempShape.Length);
            return true;
        }
        else
        {
            result = null;
            return false;
        }
    }

    public MArray SwapAxis(int dim0 = 0, int dim1 = 1)
    {
        var newArray = View();
        if (dim0 < 0) dim0 += Rank;
        if (dim1 < 0) dim1 += Rank;
        Span<int> metadata = newArray.Metadata.AsSpan();
        (metadata[dim0], metadata[dim1]) =
            (metadata[dim1], metadata[dim0]);
        (metadata[Rank + dim0], metadata[Rank + dim1]) =
            (metadata[Rank + dim1], metadata[Rank + dim0]);
        return newArray;
    }

    public MArray Squeeze()
    {
        Span<SingleIndice> indices = stackalloc SingleIndice[Rank];
        int newRank = 0;
        var lengths = Lengths;
        var strides = Strides;
        for (int i = 0; i < indices.Length; i++)
        {
            var length = lengths[i];
            var stride = strides[i];
            if (length != 1)
            {
                indices[newRank] = new(length, stride);
                newRank++;
            }
        }
        if (newRank == Rank)
        {
            return this; // No change needed
        }
        else
        {
            int[] newShape = new int[newRank * 2];
            var newLengths = newShape.AsSpan(0, newRank);
            var newStrides = newShape.AsSpan(newRank, newRank);
            for (int i = 0; i < newRank; i++)
            {
                newLengths[i] = indices[i].Length;
                newStrides[i] = indices[i].Stride;
            }
            return new MArray(Data, Offset, newShape, newRank);
        }
    }

    public MArray SqueezeDimension(int dim)
    {
        if (dim < 0 || dim >= Rank)
            throw new ArgumentOutOfRangeException(nameof(dim),
                $"The dim value {dim} is out of range [0, {Rank})");
        var length = Lengths[dim];
        if (length != 1)
            throw new ArgumentException(
                $"The dimension {dim} has length {length}, " +
                "which cannot be squeezed.", nameof(dim));
        int newRank = Rank - 1;
        int[] newShape = new int[newRank * 2];
        Array.Copy(Metadata, newShape, dim);
        //Skip Lengths[dim]
        Array.Copy(Metadata, dim + 1, newShape, dim, Rank - 1);
        //Skip Strides[dim]
        Array.Copy(Metadata, Rank + dim + 1, newShape,
            newRank + dim, Rank - dim - 1);
        return new MArray(Data, Offset, newShape, newRank);
    }

    public MArray SliceFirstDim(int index)
    {
        int[] metadata = new int[(Rank - 1) * 2];
        int length0 = Lengths[0];
        int stride0 = Strides[0];
        ThrowUtils.ThrowIfNotInRange_CheckNegative
            (ref index, length0,
            $"The index {index} is out of bounds for the first dimension with length {length0}.");
        int offset = Offset + index * stride0;
        for (int i = 1; i < Rank; i++)
        {
            metadata[i - 1] = Lengths[i];
            metadata[Rank + i - 2] = Strides[i];
        }
        return new MArray(Data, offset, metadata, Rank - 1);
    }

    public void SliceFirstDim(int index, MArray orig)
    {
        int[] metadata = new int[(Rank - 1) * 2];
        int length0 = Lengths[0];
        int stride0 = Strides[0];
        ThrowUtils.ThrowIfNotInRange_CheckNegative
            (ref index, length0,
            $"The index {index} is out of bounds for the first dimension with length {length0}.");
        int offset = Offset + index * stride0;
        for (int i = 1; i < Rank; i++)
        {
            metadata[i - 1] = Lengths[i];
            metadata[Rank + i - 2] = Strides[i];
        }
        BlasProvider.Copy(orig, new MArray(Data, offset, metadata, Rank - 1));
    }

    public MArray Transpose(params ReadOnlySpan<int> dims)
    {
        var newArray = View();
        var head = newArray.Metadata;
        Span<bool> used = stackalloc bool[Rank];
        if (dims.Length == Rank)
        {
            var lengths = Lengths;
            var strides = Strides;
            var newLengths = newArray.Metadata.AsSpan(0, Rank);
            var newStrides = newArray.Metadata.AsSpan(Rank);
            for (int i = 0; i < Rank; i++)
            {
                var dim = dims[i];
                if (dim < 0 || dim >= Rank)
                    throw new ArgumentOutOfRangeException(nameof(dims),
                        $"The dim value {dim} is out of range [0, {Rank})");
                if (used[dim])
                    throw new ArgumentException(
                        "The axis must not contain duplicate values",
                        nameof(dims));
                used[dim] = true;
                newLengths[i] = lengths[dim];
                newStrides[i] = strides[dim];
            }
            return newArray;
        }
        else if (dims.Length == 0)
        {
            var newLengths = newArray.Metadata.AsSpan(0, Rank);
            var newStrides = newArray.Metadata.AsSpan(Rank);
            newLengths.Reverse();
            newStrides.Reverse();
            return newArray;
        }
        throw new ArgumentException("The length of the axis must be equal to the rank of the array or empty", nameof(dims));
    }

    public MArray View()
    {
        return new MArray(Data,
            Offset, Metadata.AsSpan().ToArray(), Rank);
    }
}
