using System.Buffers;

namespace DDLA.Misc.Pools;

public struct ArrayHandle<T, TPool> : IPooledHandle<T[]> where TPool : ArrayPool<T>
{
    public TPool Pool { get; }

    public int Length { get; }

    public ArrayHandle(TPool pool, int length, out T[] value)
    {
        Pool = pool;
        Length = length;
        Value = pool.Rent(length);
        value = Value;
    }

    public readonly T[] Value { get; }

    public bool ReturnedToPool { get; private set; } = false;


    public static implicit operator T[](ArrayHandle<T, TPool> handle)
    {
        return handle.Value;
    }

    public void Return()
    {
        Value.AsSpan(0, Length).Clear();
        Pool.Return(Value);
        ReturnedToPool = true;
    }
}
