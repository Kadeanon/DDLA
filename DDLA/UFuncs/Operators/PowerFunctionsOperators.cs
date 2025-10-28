using System.Numerics;
using System.Runtime.CompilerServices;

namespace DDLA.UFuncs.Operators;

public readonly struct CubeOperator<T>
    : IUnaryOperator<T, T> where T : struct, IPowerFunctions<T>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public T Invoke(T x)
        => x * x * x;

    public bool IsVectorizable => Vector<T>.IsSupported;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector<T> Invoke(ref readonly Vector<T> x)
        => x * x * x;
}

public readonly struct PowOperator<T>
    : IBinaryOperator<T, T, T> where T : struct, IPowerFunctions<T>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public T Invoke(T x, T y)
        => T.Pow(x, y);
}

