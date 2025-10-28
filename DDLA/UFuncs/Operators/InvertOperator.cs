using System.Numerics;
using System.Runtime.CompilerServices;

namespace DDLA.UFuncs.Operators;

public struct DoubleInvertOperator : IUnaryOperator<double, double>
{
    public double one;

    public Vector<double> oneVec;

    static bool IOperator.ShouldManualInitialize => true;

    public DoubleInvertOperator()
    {
        one = 1;
        oneVec = new Vector<double>(one);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double Invoke(double x)
        => one / x;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector<double> Invoke(ref readonly Vector<double> x)
        => oneVec / x;

    static bool IOperator.IsVectorizable => true;
}
