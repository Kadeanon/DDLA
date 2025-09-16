using DDLA.UFuncs;
using DDLA.UFuncs.Operators;

namespace DDLA.Core;

public partial class MArray
{
    #region oop UFunc
    public void Apply<TAction, TIn>(TIn alpha)
            where TAction : struct, IUnaryOperator<TIn, double>
            where TIn : struct
        => UFunc.Apply<TAction, TIn>(this, alpha, new());

    public void Apply<TAction>(double alpha)
            where TAction : struct, IUnaryOperator<double, double>
        => UFunc.Apply<TAction, double>(this, alpha, new());

    public void Map<TAction>()
            where TAction : struct, IUnaryOperator<double, double>
        => UFunc.Map<TAction>(this, new());

    public void Map<TAction, TIn>(TIn alpha)
            where TAction : struct, IBinaryOperator<double, TIn, double>
            where TIn : struct
        => UFunc.Map<TAction, TIn>(this, alpha, new());

    public void Map<TAction>(double alpha)
            where TAction : struct, IBinaryOperator<double, double, double>
        => UFunc.Map<TAction, double>(this, alpha, new());

    public void Map<TAction>(MArray dest)
            where TAction : struct, IUnaryOperator<double, double>
        => UFunc.Map<TAction>(this, dest, new());

    public void Map<TAction, TIn>(TIn alpha, MArray dest)
            where TAction : struct, IBinaryOperator<double, TIn, double>
            where TIn : struct
        => UFunc.Map<TAction, TIn>(this, alpha, dest, new());

    public void Map<TAction>(double alpha, MArray dest)
            where TAction : struct, IBinaryOperator<double, double, double>
        => UFunc.Map<TAction, double>(this, alpha, dest, new());

    public void Combine<TAction>(MArray dest)
            where TAction : struct, IBinaryOperator<double, double, double>
        => UFunc.Combine<TAction>(this, dest, new());

    public void Combine<TAction, TIn>(TIn alpha, MArray dest)
            where TAction : struct, ITernaryOperator<double, TIn, double, double>
            where TIn : struct
        => UFunc.Combine<TAction, TIn>(this, alpha, dest, new());

    public void Combine<TAction>(double alpha, MArray dest)
            where TAction : struct, ITernaryOperator<double, double, double, double>
        => UFunc.Combine<TAction, double>(this, alpha, dest, new());
    #endregion oop UFunc

    #region operators
    public static MArray operator +(MArray left, MArray right)
    {
        var result = right.Clone();
        left.Combine<AddOperator<double>>(result);
        return result;
    }

    public static MArray operator -(MArray left, MArray right)
    {
        var result = right.Clone();
        left.Combine<SubtractOperator<double>>(result);
        return result;
    }

    public static MArray operator +(MArray left, double right)
    {
        var result = left.Clone();
        result.Map<AddOperator<double>>(right);
        return result;
    }

    public static MArray operator +(double left, MArray right)
        => right + left;

    public static MArray operator -(MArray left, double right)
    {
        var result = left.Clone();
        result.Map<SubtractOperator<double>>(right);
        return result;
    }

    public static MArray operator -(double left, MArray right)
    {
        var result = right.Clone();
        result.Map<SwappedOp<SubtractOperator<double>, double, double>>(left);
        return result;
    }

    public static MArray operator *(MArray left, double right)
    {
        var result = left.Clone();
        result.Map<MultiplyOperator<double>>(right);
        return result;
    }

    public static MArray operator *(double left, MArray right)
        => right * left;

    public static MArray operator /(MArray left, double right)
    {
        var result = left.Clone();
        result.Map<DivideOperator<double>>(right);
        return result;
    }
    #endregion operators

    #region self operations

    public MArray ScaledBy(double alpha)
    {
        if (alpha is 0 or -0)
            Apply<IdentityOperator<double>>(alpha);
        else if (alpha is not 1)
            Map<MultiplyOperator<double>>(alpha);
        return this;
    }

    public MArray AddedBy(MArray other)
    {
        other.Combine<AddOperator<double>>(this);
        return this;
    }

    public MArray AddedBy(double alpha)
    {
        if (alpha != 1)
            Map<AddOperator<double>>(alpha);
        return this;
    }

    public MArray AddedByScaled(double alpha, MArray other)
    {
        other.Combine<MultiplyAddOperator<double>>(alpha, this);
        return this;
    }

    public MArray SubtractedBy(MArray other)
    {
        other.Combine<SwappedOp<SubtractOperator<double>, double, double>>
            (this);
        return this;
    }

    public MArray SubtractedBy(double alpha)
    {
        if (alpha != 1)
            Map<SubtractOperator<double>>(alpha);
        return this;
    }

    public MArray AssignedBy(MArray other)
    {
        other.Map<IdentityOperator<double>>(this);
        return this;
    }

    public MArray AssignedBy(double alpha)
    {
        Apply<IdentityOperator<double>, double>(alpha);
        return this;
    }
    #endregion self operations
}
