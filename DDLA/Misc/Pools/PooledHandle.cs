namespace DDLA.Misc.Pools;

public interface IPooledHandle<T> : IDisposable
{
    void IDisposable.Dispose()
    {
        if (!ReturnedToPool)
        {
            Return();
            return;
        }
    }

    public bool ReturnedToPool { get; }
    public void Return();
}
