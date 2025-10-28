using scalar = double;
using tensor = DDLA.Core.MArray;
using DDLA.Misc.Flags;
using DDLA.UFuncs;
using DDLA.UFuncs.Operators;

namespace DDLA.BLAS.Managed;

public static partial class BlasProvider
{
    /// <summary>
    /// B += A
    /// </summary>
    public static void Add(in tensor A, in tensor B)
        => UFunc.Combine<AddOperator<double>>
        (A, B, default);

    /// <summary>
    /// B += alpha * Trans?(A)
    /// </summary>
    public static void Axpy(scalar alpha,
        in tensor A, in tensor B)
        => UFunc.Combine<MultiplyAddOperator<double>, double>
        (A, alpha, B, default);

    /// <summary>
    /// B = Trans?(A)
    /// </summary>
    public static void Copy(in tensor A, in tensor B)
        => UFunc.Map<IdentityOperator<double>>
        (A, B, default);

    /// <summary>
    /// A = A / alpha
    /// </summary>
    public static void InvScal(scalar alpha,
        in tensor A)
    {
        ArgumentOutOfRangeException.ThrowIfEqual(alpha, 0, nameof(alpha));
        UFunc.Map<MultiplyOperator<double>, double>
            (A, 1 / alpha, default);
    }

    /// <summary>
    /// A = alpha * A
    /// </summary>
    public static void Scal(scalar alpha,
        in tensor A)
        => UFunc.Map<MultiplyOperator<double>, double>
            (A, alpha, default);

    /// <summary>
    /// B = alpha * Trans?(A)
    /// </summary>
    public static void Scal2(double alpha,
        in tensor A, in tensor B)
        => UFunc.Map<MultiplyOperator<double>, double>
            (A, alpha, B, default);

    /// <summary>
    /// B = alpha
    /// </summary>
    public static void Set(scalar alpha,
        in tensor A)
        => UFunc.Apply<IdentityOperator<double>, double>
            (A, alpha, default);

    /// <summary>
    /// B += alpha
    /// </summary>
    public static void Shift(scalar alpha,
        in tensor A)
        => UFunc.Map<AddOperator<double>, double>
            (A, alpha, default);

    /// <summary>
    /// B -= Trans?(A)
    /// </summary>
    public static void Sub(in tensor A, in tensor B)
        => UFunc.Combine<SubtractOperator<double>>
        (A, B, default);
}
