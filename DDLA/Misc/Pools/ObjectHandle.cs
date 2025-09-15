using Microsoft.Extensions.ObjectPool;

namespace DDLA.Misc.Pools;

public struct ObjectHandle<T, TPool> : IPooledHandle<T>
    where TPool : ObjectPool<T> where T : class
{
    public TPool Pool { get; }
    public T Value { get; private set; }
    public bool ReturnedToPool { get; private set; } = false;

    public ObjectHandle(TPool pool, out T value)
    {
        Pool = pool;
        Value = pool.Get();
        value = Value;
    }

    public static implicit operator T(ObjectHandle<T, TPool> handle)
    {
        return handle.Value;
    }

    public void Return()
    {
        Pool.Return(Value);
        ReturnedToPool = true;
    }
}
