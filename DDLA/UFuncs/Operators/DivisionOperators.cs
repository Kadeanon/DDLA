using DDLA.UFuncs;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace DDLA.UFuncs.Operators;

public readonly struct DivideOperator<T>
    : IBinaryOperator<double, double, double>
    where T : struct, IMultiplyOperators<T, T, T>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double Invoke(double x, double y)
        => x / y;

    public static bool IsVectorizable => true;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector<double> Invoke(ref readonly Vector<double> x, ref readonly Vector<double> y)
        => x / y;
}

public readonly struct CheckedDivideOperator<T>
    : IBinaryOperator<double, double, double>
    where T : struct, IMultiplyOperators<T, T, T>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double Invoke(double x, double y)
        => checked(x / y);
}
