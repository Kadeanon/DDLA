using System.Numerics;
using System.Runtime.CompilerServices;

namespace DDLA.UFuncs.Operators;

public readonly struct IdentityOperator<T>
    : IUnaryOperator<T, T>
    where T : struct
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public T Invoke(T x)
        => x;

    public static bool IsVectorizable => Vector<T>.IsSupported;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector<T> Invoke(ref readonly Vector<T> x)
        => x;
}