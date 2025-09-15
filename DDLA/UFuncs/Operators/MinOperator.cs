using DDLA.UFuncs;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace DDLA.UFuncs.Operators;

public struct MinAggregationOperator<T>
    : IAggregationOperator<T, T>
    where T : struct, INumber<T>, IMinMaxValue<T>
{
    public MinAggregationOperator()
    {
        Result = Seed;
    }

    public static T Seed => T.MaxValue;

    public T Result { get; private set; }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Combine(ref readonly Vector<T> added)
    {
        for (var index = 0; index < Vector<T>.Count; index++)
            Result = T.Min(Result, added[index]);
    }

    public void Combine(T added)
        => Result = T.Min(Result, added);

    public static bool IsVectorizable => Vector<T>.IsSupported;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector<T> Invoke(ref readonly Vector<T> x, ref readonly Vector<T> y)
        => Vector.ConditionalSelect(Vector.Equals(x, x),
                Vector.ConditionalSelect(Vector.Equals(y, y),
                    Vector.ConditionalSelect(Vector.Equals(x, y),
                        Vector.ConditionalSelect(Vector.LessThan(x, Vector<T>.Zero), x, y),
                        Vector.Min(x, y)),
                    y),
                x);
}

public struct MinNumberAggregationOperator<T>
    : IAggregationOperator<T, T>
    where T : struct, INumber<T>, IMinMaxValue<T>
{
    public MinNumberAggregationOperator()
    {
        Result = Seed;
    }

    public static T Seed => T.MaxValue;

    public T Result { get; private set; }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Combine(ref readonly Vector<T> added)
    {
        for (var index = 0; index < Vector<T>.Count; index++)
            Result = T.Min(Result, added[index]);
    }

    public void Combine(T added)
        => Result = T.MinNumber(Result, added);

    public static bool IsVectorizable => Vector<T>.IsSupported;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector<T> Invoke(ref readonly Vector<T> x, ref readonly Vector<T> y)
        => Vector.MinNumber(x, y);
}

public readonly struct MinOperator<T>
    : IBinaryOperator<T, T, T>
    where T : struct, INumber<T>, IMinMaxValue<T>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public T Invoke(T x, T y)
        => T.Min(x, y);

    public static bool IsVectorizable => Vector<T>.IsSupported;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector<T> Invoke(ref readonly Vector<T> x, ref readonly Vector<T> y)
        => Vector.ConditionalSelect(Vector.Equals(x, x),
                Vector.ConditionalSelect(Vector.Equals(y, y),
                    Vector.ConditionalSelect(Vector.Equals(x, y),
                        Vector.ConditionalSelect(Vector.LessThan(x, Vector<T>.Zero), x, y),
                        Vector.Min(x, y)),
                    y),
                x);
}

public readonly struct MinNumberOperator<T>
    : IBinaryOperator<T, T, T>
    where T : struct, INumber<T>, IMinMaxValue<T>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public T Invoke(T x, T y)
        => T.MinNumber(x, y);

    public static bool IsVectorizable => Vector<T>.IsSupported;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector<T> Invoke(ref readonly Vector<T> x, ref readonly Vector<T> y)
        => Vector.MinNumber(x, y);
}