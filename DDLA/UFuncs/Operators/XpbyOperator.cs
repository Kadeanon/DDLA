using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace DDLA.UFuncs.Operators;

public struct DoubleXpbyOperator : ITernaryOperator<double, double, double, double>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double Invoke(double x, double a, double y)
        => x + a * y;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector<double> Invoke(ref readonly Vector<double> x,
        ref readonly Vector<double> a, ref readonly Vector<double> y)
    {
        if (Fma.IsSupported && Vector<double>.Count == 2)
        {
            return Fma.MultiplyAdd(
                a.AsVector128(), y.AsVector128(), x.AsVector128()).AsVector();
        }
        else if (Fma.IsSupported && Vector<double>.Count == 4)
        {
            return Fma.MultiplyAdd(
                a.AsVector256(), y.AsVector256(), x.AsVector256()).AsVector();
        }
        else
        {
            return x + a * y;
        }
    }
}
