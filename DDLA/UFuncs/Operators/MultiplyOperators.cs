using System.Numerics;
using System.Runtime.CompilerServices;

namespace DDLA.UFuncs.Operators;

public readonly struct MultiplyOperator<T>
    : IBinaryOperator<T, T, T>
    where T : struct, IMultiplyOperators<T, T, T>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public T Invoke(T x, T y)
        => x * y;

    public static bool IsVectorizable => true;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector<T> Invoke(ref readonly Vector<T> x, ref readonly Vector<T> y)
        => x * y;
}

public readonly struct CheckedMultiplyOperator<T>
    : IBinaryOperator<T, T, T>
    where T : struct, IMultiplyOperators<T, T, T>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public T Invoke(T x, T y)
        => checked(x * y);
}
