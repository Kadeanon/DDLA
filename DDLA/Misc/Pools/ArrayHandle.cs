using System.Buffers;

namespace DDLA.Misc.Pools;

public struct ArrayHandle<T, TPool> : IPooledHandle<T[]> where TPool : ArrayPool<T>
{
    public TPool Pool { get; }

    public int Length { get; }

    public bool ClearBeforeReturn { get; }

    public ArrayHandle(TPool pool, int length, out T[] value, 
        bool clearBeforeReturn = true)
    {
        Pool = pool;
        Length = length;
        Value = pool.Rent(length);
        value = Value;
        ClearBeforeReturn = clearBeforeReturn;
    }

    public readonly T[] Value { get; }

    public bool ReturnedToPool { get; private set; } = false;


    public static implicit operator T[](ArrayHandle<T, TPool> handle)
    {
        return handle.Value;
    }

    public void Return()
    {
        if(ClearBeforeReturn)
            Value.AsSpan(0, Length).Clear();
        Pool.Return(Value);
        ReturnedToPool = true;
    }
}
