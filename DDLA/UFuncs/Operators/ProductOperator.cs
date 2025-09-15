using DDLA.UFuncs;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace DDLA.UFuncs.Operators;

public struct ProductOperator<T>
    : IAggregationOperator<T, T>
    where T : struct, IMultiplyOperators<T, T, T>,
    IMultiplicativeIdentity<T, T>
{
    public ProductOperator()
    {
        Result = Seed;
    }

    public static T Seed => T.MultiplicativeIdentity;

    public T Result { get; private set; }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Combine(ref readonly Vector<T> added)
    {
        for (var index = 0; index < Vector<T>.Count; index++)
            Result *= added[index];
    }

    public void Combine(T added)
        => Result *= added;

    public static bool IsVectorizable => Vector<T>.IsSupported;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<T> Invoke(ref readonly Vector<T> x, ref readonly Vector<T> y)
        => x * y;
}