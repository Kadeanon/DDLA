using DDLA.UFuncs;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace DDLA.UFuncs.Operators;

public struct MaxAggregationOperator<T>
    : IAggregationOperator<T, T>
    where T : struct, INumber<T>, IMinMaxValue<T>
{
    public MaxAggregationOperator()
    {
        Result = Seed;
    }

    public static T Seed => T.MinValue;

    public T Result { get; private set; }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Combine(ref readonly Vector<T> added)
    {
        for (var index = 0; index < Vector<T>.Count; index++)
            Result = T.Max(Result, added[index]);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Combine(T added)
        => Result = T.Max(Result, added);

    public static bool IsVectorizable => Vector<T>.IsSupported;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector<T> Invoke(ref readonly Vector<T> x, ref readonly Vector<T> y)
        => Vector.ConditionalSelect(Vector.Equals(x, x),
                Vector.ConditionalSelect(Vector.Equals(y, y),
                    Vector.ConditionalSelect(Vector.Equals(x, y),
                        Vector.ConditionalSelect(Vector.LessThan(x, Vector<T>.Zero), y, x),
                        Vector.Max(x, y)),
                    y),
                x);
}

public struct MaxNumberAggregationOperator<T>
    : IAggregationOperator<T, T>
    where T : struct, INumber<T>, IMinMaxValue<T>
{
    public MaxNumberAggregationOperator()
    {
        Result = Seed;
    }

    public static T Seed => T.MinValue;

    public T Result { get; private set; }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Combine(ref readonly Vector<T> added)
    {
        for (var index = 0; index < Vector<T>.Count; index++)
            Result = T.MaxNumber(Result, added[index]);
    }

    public void Combine(T added)
        => Result = T.MaxNumber(Result, added);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector<T> Invoke(ref readonly Vector<T> x, ref readonly Vector<T> y)
        => Vector.MaxNumber(x, y);
}

public readonly struct MaxOperator<T>
    : IBinaryOperator<T, T, T>
    where T : struct, INumber<T>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public T Invoke(T x, T y)
        => T.Max(x, y);

    public static bool IsVectorizable => Vector<T>.IsSupported;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector<T> Invoke(ref readonly Vector<T> x, ref readonly Vector<T> y)
        => Vector.ConditionalSelect(Vector.Equals(x, x),
                Vector.ConditionalSelect(Vector.Equals(y, y),
                    Vector.ConditionalSelect(Vector.Equals(x, y),
                        Vector.ConditionalSelect(Vector.LessThan(x, Vector<T>.Zero), y, x),
                        Vector.Max(x, y)),
                    y),
                x);
}

public readonly struct MaxNumberOperator<T>
    : IBinaryOperator<T, T, T>
    where T : struct, INumber<T>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public T Invoke(T x, T y)
        => T.MaxNumber(x, y);

    public static bool IsVectorizable => Vector<T>.IsSupported;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector<T> Invoke(ref readonly Vector<T> x, ref readonly Vector<T> y)
        => Vector.MaxNumber(x, y);
}