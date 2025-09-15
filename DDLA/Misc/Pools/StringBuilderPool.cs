using Microsoft.Extensions.ObjectPool;
using System.Text;

namespace DDLA.Misc.Pools;

public class StringBuilderPool(StringBuilderPooledObjectPolicy policy) : DefaultObjectPool<StringBuilder>(policy, policy.MaximumRetainedCapacity)
{
    public readonly static StringBuilderPool Shared = new(new() { MaximumRetainedCapacity = Environment.ProcessorCount * 2 });

    public static ObjectHandle<StringBuilder, StringBuilderPool> Borrow(out StringBuilder sb) => Shared.GetHandle(out sb);

    public ObjectHandle<StringBuilder, StringBuilderPool> GetHandle(out StringBuilder sb)
    {
        var handle = new ObjectHandle<StringBuilder, StringBuilderPool>(this, out sb);
        return handle;
    }
}
