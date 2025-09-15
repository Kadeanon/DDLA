using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace DDLA.UFuncs.Operators
{
    public readonly struct SquareOperator<T>
    : IUnaryOperator<T, T>
    where T : struct, IMultiplyOperators<T, T, T>
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public T Invoke(T x)
            => x * x;

        public static bool IsVectorizable => Vector<T>.IsSupported;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Vector<T> Invoke(ref readonly Vector<T> x)
            => x * x;
    }
}
