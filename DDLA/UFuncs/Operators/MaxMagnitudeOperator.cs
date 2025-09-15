using DDLA.UFuncs;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace DDLA.UFuncs.Operators;

public struct MaxMagnitudeAggregationOperator<T>
    : IAggregationOperator<T, T>
    where T : struct, INumberBase<T>, IMinMaxValue<T>
{
    public MaxMagnitudeAggregationOperator()
    {
        Result = Seed;
    }

    public static T Seed => T.MinValue;

    public T Result { get; private set; }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Combine(ref readonly Vector<T> added)
    {
        for (var index = 0; index < Vector<T>.Count; index++)
            Result = T.MaxMagnitude(Result, added[index]);
    }

    public void Combine(T added)
        => Result = T.MaxMagnitude(Result, added);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector<T> Invoke(ref readonly Vector<T> x, ref readonly Vector<T> y)
    {
        var xMag = Vector.Abs(x);
        var yMag = Vector.Abs(y);
        return Vector.ConditionalSelect(Vector.Equals(x, x),
                Vector.ConditionalSelect(Vector.Equals(y, y),
                    Vector.ConditionalSelect(Vector.Equals(xMag, yMag),
                        Vector.ConditionalSelect(Vector.LessThan(x, Vector<T>.Zero), y, x),
                        Vector.ConditionalSelect(Vector.GreaterThan(xMag, yMag), x, y)),
                    y),
                x);
    }
}

public struct MaxMagnitudeNumberAggregationOperator<T>
    : IAggregationOperator<T, T>
    where T : struct, INumberBase<T>, IMinMaxValue<T>
{
    public MaxMagnitudeNumberAggregationOperator()
    {
        Result = Seed;
    }

    public static T Seed => T.MinValue;

    public T Result { get; private set; }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Combine(ref readonly Vector<T> added)
    {
        for (var index = 0; index < Vector<T>.Count; index++)
            Result = T.MaxMagnitudeNumber(Result, added[index]);
    }

    public void Combine(T added)
        => Result = T.MaxMagnitudeNumber(Result, added);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector<T> Invoke(ref readonly Vector<T> x, ref readonly Vector<T> y)
    {
        var xMag = Vector.Abs(x);
        var yMag = Vector.Abs(y);
        return
            Vector.ConditionalSelect(Vector.Equals(xMag, yMag),
                Vector.ConditionalSelect(Vector.LessThan(x, Vector<T>.Zero), y, x),
                Vector.ConditionalSelect(Vector.GreaterThan(xMag, yMag), x, y));
    }
}

public readonly struct MaxMagnitudeOperator<T>
    : IBinaryOperator<T, T, T>
    where T : struct, INumberBase<T>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public T Invoke(T x, T y)
        => T.MaxMagnitude(x, y);

}

public readonly struct MaxMagnitudeNumberOperator<T>
    : IBinaryOperator<T, T, T>
    where T : struct, INumberBase<T>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public T Invoke(T x, T y)
        => T.MaxMagnitudeNumber(x, y);
}