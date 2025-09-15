using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace DDLA.UFuncs.Operators;

public readonly struct AddOperator<T> : IBinaryOperator<T, T, T>
    where T : struct, IAdditionOperators<T, T, T>
{
    public T Invoke(T x, T y)
    {
        return x + y;
    }

    public static bool IsVectorizable => Vector<T>.IsSupported;

    public Vector<T> Invoke(ref readonly Vector<T> x,
        ref readonly Vector<T> y)
        => x + y;
}
