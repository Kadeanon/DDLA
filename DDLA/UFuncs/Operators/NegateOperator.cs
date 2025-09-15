using System.Numerics;

namespace DDLA.UFuncs.Operators
{
    public readonly struct NegateOperator<T> : IUnaryOperator<T, T>
        where T : struct, IUnaryNegationOperators<T, T>
    {
        public T Invoke(T x)
        {
            return -x;
        }

        public static bool IsVectorizable => true;

        public Vector<T> Invoke(ref readonly Vector<T> x)
            => -x;
    }
}
