using DDLA.Misc;
using DDLA.Utilities;

namespace DDLA.Core;

public partial class MArray
{
    public ref double this[params ReadOnlySpan<int> indexes]
    {
        get
        {
            int rank = Rank;
            int numIndex = indexes.Length;
            ArgumentOutOfRangeException.ThrowIfNotEqual(numIndex, rank,
                $"The number of indexs ({numIndex}) must be " +
                $"equal to the rank of the array ({rank}).");
            Span<Range> newRanges = stackalloc Range[rank];
            int offset = Offset;
            int i = 0;
            for (; i < rank; i++)
            {
                var dimOffset = indexes[i];
                var dimTotalLength = Lengths[i];
                ThrowUtils.ThrowIfNotInRange_LeftClosedRightOpen
                    (dimOffset, 0, dimTotalLength,
                    $"The index {dimOffset} is out of bounds " +
                    $"for dimension {i} with length {dimTotalLength}.");
                offset += dimOffset * Strides[i];
            }
            return ref Data[offset];
        }
    }

    public ref double this[params ReadOnlySpan<Index> index]
    {
        get
        {
            var rank = Rank;
            var numIndex = index.Length;
            ArgumentOutOfRangeException.ThrowIfNotEqual(numIndex, rank,
                $"The number of indexs ({numIndex}) must be " +
                $"equal to the rank of the array ({rank}).");
            Span<Range> newRanges = stackalloc Range[rank];
            var offset = Offset;
            var i = 0;
            for (; i < rank; i++)
            {
                var dimIndex = index[i];
                var dimTotalLength = Lengths[i];
                var dimOffset =
                    dimIndex.GetOffset(dimTotalLength);
                ThrowUtils.ThrowIfNotInRange_LeftClosedRightOpen
                    (dimOffset, 0, dimTotalLength,
                    $"The index {dimOffset} is out of bounds " +
                    $"for dimension {i} with length {dimTotalLength}.");
                offset += dimOffset * Strides[i];
            }
            return ref Data[offset];
        }
    }

    public MArray this[params ReadOnlySpan<Range> ranges]
    {
        get => GetData(ranges);
        set => SetData(value, ranges);
    }

    public MArray this[params ReadOnlySpan<Slice> slices]
    {
        get => GetData(slices);
        set => SetData(value, slices);
    }

    public MArray GetData(params ReadOnlySpan<Range> ranges)
    {
        int rank = Rank;
        int numRanges = ranges.Length;
        ArgumentOutOfRangeException.ThrowIfGreaterThan(numRanges, rank,
            $"The number of ranges ({numRanges}) must be " +
            $"less than or equal to the rank of the array ({rank}).");
        int offset = Offset;
        int[] metadata = new int[rank * 2];
        int i = 0;
        for (; i < ranges.Length; i++)
        {
            var dimRange = ranges[i];
            var dimTotalLength = Lengths[i];
            var (dimStart, dimLength) = ThrowUtils.ThrowIfNotInRange_CheckNRange
                (dimRange, dimTotalLength, nameof(ranges));
            offset += dimStart * Strides[i];
            metadata[i] = dimLength;
            metadata[i + rank] = Strides[i];
        }
        for (; i < this.Rank; i++)
        {
            metadata[i] = Lengths[i];
            metadata[i + rank] = Strides[i];
        }
        return new MArray(Data, offset, metadata, rank);
    }

    public void SetData(MArray value, params ReadOnlySpan<Range> ranges)
    {
        GetData(ranges).AssignedBy(value);
    }

    public void SetData(double value, params ReadOnlySpan<Range> ranges)
    {
        var array = GetData(ranges);
        array.AssignedBy(value);
    }

    public MArray GetData(params ReadOnlySpan<Slice> slices)
    {
        var rank = Rank;
        var newRank = rank;
        var numSlices = slices.Length;
        ArgumentOutOfRangeException.ThrowIfGreaterThan(numSlices, rank,
            $"The number of ranges ({numSlices}) must be " +
            $"less than or equal to the rank of the array ({rank}).");
        Span<bool> indexers = stackalloc bool[slices.Length];
        Span<Range> newRanges = stackalloc Range[rank];
        var offset = Offset;
        Span<int> metadata = stackalloc int[rank * 2];
        var i = 0;
        var j = 0;
        for (; i < numSlices; i++)
        {
            var slice = slices[i];
            if (slice.IsIndex)
            {
                var dimOffset = ThrowUtils.ThrowIfNotInRange_CheckNIndex
                    (slice.Index, Lengths[i], nameof(slice));
                offset += dimOffset * Strides[i];
            }
            else
            {
                var (dimOffset, dimLength) = ThrowUtils.ThrowIfNotInRange_CheckNRange
                    (slice.Range, Lengths[i], nameof(slice));
                offset += dimOffset * Strides[i];
                metadata[j] = dimLength;
                metadata[j + rank] = Strides[i];
                j++;
            }
        }
        if (j < i)
        {
            newRank -= i - j;
            for (int j2 = 0; j2 < j; j2++)
            {
                metadata[j2 + newRank] = metadata[j2 + rank];
            }
            rank = newRank;
        }
        for (; i < rank; i++)
        {
            metadata[j] = Lengths[i];
            metadata[j + rank] = Strides[i + rank];
            j++;
        }
        return new MArray(Data, offset, [.. metadata[..(newRank * 2)]], newRank);
    }

    public void SetData(MArray value, params ReadOnlySpan<Slice> slices)
    {
        var array = GetData(slices);
        array.AssignedBy(value);
    }
}
