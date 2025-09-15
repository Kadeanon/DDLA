using DDLA.UFuncs;
using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace DDLA.UFuncs.Operators;

public struct SumOperator<T>
    : IAggregationOperator<T, T>
    where T : struct, 
    IAdditionOperators<T, T, T>, IAdditiveIdentity<T, T>
{
    public SumOperator()
    {
        Result = Seed;
    }

    public static T Seed => T.AdditiveIdentity;

    public T Result { get; private set; }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Combine(ref readonly Vector<T> added)
    {
        Result += Vector.Sum(added);
    }

    public void Combine(T added)
        => Result += added;

    public static bool IsVectorizable => Vector<T>.IsSupported;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<T> Invoke(ref readonly Vector<T> x, ref readonly Vector<T> y)
        => x + y;
}