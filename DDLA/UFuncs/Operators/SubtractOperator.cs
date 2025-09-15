using System.Numerics;

namespace DDLA.UFuncs.Operators;

public readonly struct SubtractOperator<T> : IBinaryOperator<T, T, T>
    where T : struct, ISubtractionOperators<T, T, T>
{
    public T Invoke(T x, T y)
    {
        return x - y;
    }

    public static bool IsVectorizable => Vector<T>.IsSupported;

    public Vector<T> Invoke(ref readonly Vector<T> x,
        ref readonly Vector<T> y)
        => x - y;
}
