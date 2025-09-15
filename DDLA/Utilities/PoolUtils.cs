using DDLA.Misc.Pools;
using System.Buffers;
using System.Runtime.CompilerServices;

namespace DDLA.Utilities;

public static class PoolUtils
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ArrayHandle<T, ArrayPool<T>> Borrow<T>(int length, out T[] value)
    {
        return new(ArrayPool<T>.Shared, length, out value);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ArrayHandle<T, ArrayPool<T>> Borrow<T>(ArrayPool<T> pool, int length, out T[] value)
    {
        return new(pool, length, out value);
    }

    public static ArrayHandle<T, ArrayPool<T>>? HeapedSpanBorrow<T>(ArrayPool<T> pool, int length, ref Span<T> value)
    {
        if (length > value.Length)
        {
            var handle = new ArrayHandle<T, ArrayPool<T>>(pool, length, out var array);
            var span = array.AsSpan(0, length);
            span.Clear();
            value = span;
            return handle;
        }
        else
        {
            value = value[..length];
            return null;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ArrayHandle<T, ArrayPool<T>>? HeapedSpanBorrow<T>(int length, ref Span<T> value)
    {
        return HeapedSpanBorrow(ArrayPool<T>.Shared, length, ref value);
    }

}
