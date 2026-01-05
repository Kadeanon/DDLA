using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace DDLA.UFuncs.Operators;

public readonly struct GuardSquareOperator<T> : IUnaryOperator<T, T>
    where T: struct, INumberBase<T>
{
    public T ScalingFactor { get; }

    public Vector<T> ScalingFactorVec { get; }

    public GuardSquareOperator(T maxVal)
    {
        ScalingFactor = T.One / maxVal;
        ScalingFactorVec = new Vector<T>(ScalingFactor);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    T IUnaryOperator<T, T>.Invoke(T x)
    {
        var temp = ScalingFactor * x;
        return temp * temp;
    }

    static bool IOperator.IsVectorizable => Vector<T>.IsSupported;


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    Vector<T> IUnaryOperator<T, T>.Invoke(ref readonly Vector<T> x)
    {
        var temp = ScalingFactorVec * x;
        return temp * temp;
    }
}
