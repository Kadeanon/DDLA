using System.Numerics;
using System.Runtime.CompilerServices;

namespace DDLA.UFuncs.Operators;

public readonly struct CbrtOperator<T>
    : IUnaryOperator<T, T> where T : struct, IRootFunctions<T>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public T Invoke(T x)
        => T.Cbrt(x);
}

public readonly struct HypotOperator<T>
    : IBinaryOperator<T, T, T> where T : struct, IRootFunctions<T>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public T Invoke(T x, T y)
        => T.Hypot(x, y);
}

public readonly struct RootNOperator<T>
    : IBinaryOperator<T, int, T> where T : struct, IRootFunctions<T>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public T Invoke(T x, int y)
        => T.RootN(x, y);
}

public readonly struct SqrtOperator<T>
    : IUnaryOperator<T, T> where T : struct, IRootFunctions<T>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public T Invoke(T x)
        => T.Sqrt(x);

    public bool IsVectorizable => Vector<T>.IsSupported;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector<T> Invoke(ref readonly Vector<T> x)
        => Vector.SquareRoot(x);
}