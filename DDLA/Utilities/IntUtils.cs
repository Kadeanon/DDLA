using DDLA.Utilities;

namespace DDLA.Utilities;

public static class IntUtils
{
    public static bool Is64Bit => Environment.Is64BitProcess;

    public static int Product(this ReadOnlySpan<int> values, bool oneWhenEmpty = false)
    {
        if (values.IsEmpty)
        {
            return oneWhenEmpty ? 1 : 0;
        }
        int product = 1;
        foreach (var value in values)
        {
            product *= value;
        }
        return product;
    }

    public static int Dot(this ReadOnlySpan<int> left, ReadOnlySpan<int> right)
    {
        if (left.Length != right.Length)
        {
            throw new ArgumentException("The length of the two spans must be equal.");
        }
        int sum = 0;
        for (int i = 0; i < left.Length; i++)
        {
            sum += left[i] * right[i];
        }
        return sum;
    }

    public static int CalculateTotalLength(ReadOnlySpan<int> lengths, ReadOnlySpan<int> strides)
    {
        if (lengths.Length != strides.Length)
        {
            throw new ArgumentException("The length of the two spans must be equal.");
        }
        int sum = 0;
        int index = 0;
        foreach (var value in lengths)
        {
            sum += (value - 1) * strides[index];
            index++;
        }
        return sum;
    }

    public static int IncrementIndexLeft(int curIndex, Span<int> curIndexes, scoped ReadOnlySpan<int> length)
    {
        if (curIndex < 0)
            return int.MinValue;
        curIndexes[curIndex] += 1;

        int max = length[curIndex];
        if (curIndexes[curIndex] < max)
            return 1;
        curIndexes[curIndex] = 0;
        return IncrementIndexLeft(curIndex - 1, curIndexes, length) + 1;
    }

    public static int AddIndexLeft(int curIndex, Span<int> curIndexes, scoped ReadOnlySpan<int> length, int add)
    {
        if (curIndex < 0)
            return 0;
        if (add == 0)
        {
            return 0;
        }
        if (add == 1)
        {
            return IncrementIndexLeft(curIndex, curIndexes, length);
        }
        ref int current = ref curIndexes[curIndex];
        current += add;
        int max = length[curIndex];
        if (current < max)
            return 1;
        (int quotient, int remainder) = Math.DivRem(curIndexes[curIndex], max);
        current = remainder;
        return AddIndexLeft(curIndex - 1, curIndexes, length, quotient) + 1;
    }

    public static Span<T> Slice<T>(
        this Span<T> span, Range range)
    {
        var (offset, length) =
            range.GetOffsetAndLength(span.Length);
        return span.Slice(offset, length);
    }

    public static ReadOnlySpan<T> Slice<T>(
        this ReadOnlySpan<T> span, Range range)
    {
        var (offset, length) =
            range.GetOffsetAndLength(span.Length);
        return span.Slice(offset, length);
    }

    public static ref T At<T>(this Span<T> span, Index index)
    {
        var offset = index.GetOffset(span.Length);
        return ref span[offset];
    }

    public static ref readonly T At<T>(
        this ReadOnlySpan<T> span, Index index)
    {
        var offset = index.GetOffset(span.Length);
        return ref span[offset];
    }
}
