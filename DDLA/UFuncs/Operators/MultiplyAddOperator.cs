using System.Numerics;
using System.Runtime.CompilerServices;

namespace DDLA.UFuncs.Operators;

public readonly struct MultiplyAddOperator<T>
    : ITernaryOperator<double, double, double, double>
    where T : struct, IAdditionOperators<T, T, T>,
    IMultiplyOperators<T, T, T>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double Invoke(double x, double y, double z)
        => x * y + z;

    public static bool IsVectorizable => true;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector<double> Invoke(ref readonly Vector<double> x, ref readonly Vector<double> y, ref readonly Vector<double> z)
        => x * y + z;
}
