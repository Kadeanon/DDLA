using DDLA.UFuncs;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace DDLA.UFuncs.Operators;

public struct MinMagnitudeAggregationOperator<T>
    : IAggregationOperator<T, T>
    where T : struct, INumber<T>, IMinMaxValue<T>
{
    public MinMagnitudeAggregationOperator()
    {
        Result = Seed;
    }

    public static T Seed => T.MaxValue;

    public T Result { get; private set; }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Combine(ref readonly Vector<T> added)
    {
        for (var index = 0; index < Vector<T>.Count; index++)
            Result = T.MinMagnitude(Result, added[index]);
    }

    public void Combine(T added)
        => Result = T.MinMagnitude(Result, added);

    public static bool IsVectorizable => Vector<T>.IsSupported;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector<T> Invoke(ref readonly Vector<T> x, ref readonly Vector<T> y)
    {
        var xMag = Vector.Abs(x);
        var yMag = Vector.Abs(y);
        return typeof(T) == typeof(T) || typeof(T) == typeof(float) || typeof(T) == typeof(Half)
            ? Vector.ConditionalSelect(Vector.Equals(x, x),
                Vector.ConditionalSelect(Vector.Equals(y, y),
                    Vector.ConditionalSelect(Vector.Equals(yMag, xMag),
                        Vector.ConditionalSelect(Vector.LessThan(x, Vector<T>.Zero), x, y),
                        Vector.ConditionalSelect(Vector.LessThan(xMag, yMag), x, y)),
                    y),
                x)
            : Vector.ConditionalSelect(Vector.Equals(yMag, xMag),
                Vector.ConditionalSelect(Vector.LessThan(y, Vector<T>.Zero), y, x),
                Vector.ConditionalSelect(Vector.LessThan(yMag, xMag), y, x));
    }
}

public struct MinMagnitudeNumberAggregationOperator<T>
    : IAggregationOperator<T, T>
    where T : struct, INumber<T>, IMinMaxValue<T>
{
    public MinMagnitudeNumberAggregationOperator()
    {
        Result = Seed;
    }

    public static T Seed => T.MaxValue;

    public T Result { get; private set; }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Combine(ref readonly Vector<T> added)
    {
        for (var index = 0; index < Vector<T>.Count; index++)
            Result = T.MinMagnitudeNumber(Result, added[index]);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Combine(T added)
        => Result = T.MinMagnitudeNumber(Result, added);

    public static bool IsVectorizable => Vector<T>.IsSupported;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector<T> Invoke(ref readonly Vector<T> x, ref readonly Vector<T> y)
    {
        var xMag = Vector.Abs(x);
        var yMag = Vector.Abs(y);
        return
            Vector.ConditionalSelect(Vector.Equals(yMag, xMag),
                Vector.ConditionalSelect(Vector.LessThan(y, Vector<T>.Zero), y, x),
                Vector.ConditionalSelect(Vector.LessThan(yMag, xMag), y, x));
    }
}

public readonly struct MinMagnitudeOperator<T>
    : IBinaryOperator<T, T, T>
    where T : struct, INumberBase<T>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public T Invoke(T x, T y)
        => T.MinMagnitude(x, y);

    public static bool IsVectorizable => Vector<T>.IsSupported;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector<T> Invoke(ref readonly Vector<T> x, ref readonly Vector<T> y)
    {
        var xMag = Vector.Abs(x);
        var yMag = Vector.Abs(y);
        return typeof(T) == typeof(T) || typeof(T) == typeof(float) || typeof(T) == typeof(Half)
            ? Vector.ConditionalSelect(Vector.Equals(x, x),
                Vector.ConditionalSelect(Vector.Equals(y, y),
                    Vector.ConditionalSelect(Vector.Equals(yMag, xMag),
                        Vector.ConditionalSelect(Vector.LessThan(x, Vector<T>.Zero), x, y),
                        Vector.ConditionalSelect(Vector.LessThan(xMag, yMag), x, y)),
                    y),
                x)
            : Vector.ConditionalSelect(Vector.Equals(yMag, xMag),
                Vector.ConditionalSelect(Vector.LessThan(y, Vector<T>.Zero), y, x),
                Vector.ConditionalSelect(Vector.LessThan(yMag, xMag), y, x));
    }
}

public readonly struct MinMagnitudeNumberOperator<T>
    : IBinaryOperator<T, T, T>
    where T : struct, INumberBase<T>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly T Invoke(T x, T y)
        => T.MinMagnitudeNumber(x, y);

    public static bool IsVectorizable => Vector<T>.IsSupported;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly Vector<T> Invoke(ref readonly Vector<T> x, ref readonly Vector<T> y)
    {
        var xMag = Vector.Abs(x);
        var yMag = Vector.Abs(y);
        return
            Vector.ConditionalSelect(Vector.Equals(yMag, xMag),
                Vector.ConditionalSelect(Vector.LessThan(y, Vector<T>.Zero), y, x),
                Vector.ConditionalSelect(Vector.LessThan(yMag, xMag), y, x));
    }
}