using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace DDLA.UFuncs.Operators;

public readonly struct MultiplyAddOperator<T>
    : ITernaryOperator<T, T, T, T>
    where T : struct, IAdditionOperators<T, T, T>,
    IMultiplyOperators<T, T, T>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public T Invoke(T x, T y, T z)
        => x * y + z;

    public static bool IsVectorizable 
        => Vector<double>.IsSupported && Fma.IsSupported;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector<T> Invoke(ref readonly Vector<T> x, ref readonly Vector<T> y, ref readonly Vector<T> z)
    {
        if(typeof(T) == typeof(double) && IsVectorizable)
        {
            if (Vector<double>.Count == Vector256<double>.Count)
            {
                return Fma.MultiplyAdd(x.AsVector256().AsDouble(),
                    y.AsVector256().AsDouble(), 
                    z.AsVector256().AsDouble()).AsVector().As<double, T>();
            }
            else if (Vector<double>.Count == Vector128<double>.Count)
            {
                return Fma.MultiplyAdd(x.AsVector128().AsDouble(),
                    y.AsVector128().AsDouble(), 
                    z.AsVector128().AsDouble()).AsVector().As<double, T>();
            }
        }
        else if (typeof(T) == typeof(float) && IsVectorizable)
        {
            if (Vector<float>.Count == Vector256<float>.Count)
            {
                return Fma.MultiplyAdd(x.AsVector256().AsSingle(),
                    y.AsVector256().AsSingle(),
                    z.AsVector256().AsSingle()).AsVector().As<float, T>();
            }
            else if (Vector<float>.Count == Vector128<float>.Count)
            {
                return Fma.MultiplyAdd(x.AsVector128().AsSingle(),
                    y.AsVector128().AsSingle(),
                    z.AsVector128().AsSingle()).AsVector().As<float, T>();
            }
        }
        return x * y + z;
    }
}
