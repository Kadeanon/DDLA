using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace DDLA.UFuncs.Operators;

public readonly struct ReversedOp<TOperator, T1, T2, T3>(TOperator? op = null)
    : IBinaryOperator<T1, T2, T3>
    where TOperator : struct, IBinaryOperator<T2, T1, T3>
    where T1 : struct
    where T2 : struct
    where T3 : struct
{
    TOperator Operator { get; } = op.OrDefault();
    public T3 Invoke(T1 x, T2 y)
        => Operator.Invoke(y, x);

    public static bool IsVectorizable => TOperator.IsVectorizable;

    public Vector<T3> Invoke(ref readonly Vector<T1> x,
        ref readonly Vector<T2> y)
        => Operator.Invoke(in y, in x);
}

public readonly struct SwappedOp<TOperator, TIn, TOut>(TOperator op = default)
    : IBinaryOperator<TIn, TIn, TOut>
    where TOperator : struct, IBinaryOperator<TIn, TIn, TOut>
    where TIn : struct
    where TOut : struct
{
    TOperator Operator { get; } = op;
    public TOut Invoke(TIn x, TIn y)
        => Operator.Invoke(y, x);

    public static bool IsVectorizable => TOperator.IsVectorizable;

    public Vector<TOut> Invoke(ref readonly Vector<TIn> x,
        ref readonly Vector<TIn> y)
        => Operator.Invoke(in y, in x);
}
