using System.Numerics;

namespace DDLA.UFuncs.Operators;

public readonly struct AbsOperator<T> : IUnaryOperator<T, T>
    where T : struct, INumber<T>
{
    public T Invoke(T x) => T.Abs(x);
}
