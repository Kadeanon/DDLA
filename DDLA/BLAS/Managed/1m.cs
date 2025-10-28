using DDLA.Misc.Flags;
using DDLA.UFuncs;
using DDLA.UFuncs.Operators;
using matrix = DDLA.Core.MatrixView;
using scalar = double;

namespace DDLA.BLAS.Managed;

public static partial class BlasProvider
{
    /// <summary>
    /// B += Trans?(A)
    /// </summary>
    public static void Add(DiagType aDiag, UpLo aUplo, TransType aTrans,
        in matrix A, in matrix B)
    {
        var (m, n) = CheckLength(A, aTrans, B);
        if (m == 0 || n == 0) return;

        var AEffective = A;
        if (aTrans.HasFlag(TransType.OnlyTrans))
        {
            aUplo = Transpose(aUplo);
            AEffective = A.T;
        }
        var BEffective = B;
        var invoker = UFunc.OrDefault<AddOperator<scalar>>(null);
        if (B.RowStride < B.ColStride)
        {
            AEffective = AEffective.T;
            BEffective = BEffective.T;
            aUplo = Transpose(aUplo);
        }
        (m, n) = GetLengths(AEffective);

        if (aUplo is UpLo.Dense)
        {
            AEffective.Combine<AddOperator<scalar>>(BEffective, invoker);
        }
        else if (aUplo is UpLo.Lower or UpLo.Upper)
        {
            if (aUplo is UpLo.Upper)
            {
                for (int i = 0; i < m; i++)
                {
                    var start = A.DiagOffset + i;
                    if (aDiag is DiagType.Unit)
                        start++;
                    start = Math.Max(start, 0);
                    if (start >= n)
                        break;
                    var rowA = AEffective.SliceRowUncheck(i, start);
                    var rowB = BEffective.SliceRowUncheck(i, start);
                    rowA.Combine<AddOperator<scalar>>(rowB, invoker);
                }
            }
            else // if (aUplo is UpLo.Lower)
            {
                for (int i = 0; i < m; i++)
                {
                    var end = A.DiagOffset + i + 1;
                    if (aDiag is DiagType.Unit)
                        end--;
                    end = Math.Min(end, n);
                    if (end <= 0)
                        continue;
                    var rowA = AEffective.SliceRowUncheck(i, 0, end);
                    var rowB = BEffective.SliceRowUncheck(i, 0, end);
                    rowA.Combine<AddOperator<scalar>>(rowB, invoker);
                }
            }
        }
    }

    /// <summary>
    /// B += Trans?(A)
    /// </summary>
    public static void Add(in matrix A, in matrix B)
        => Add(DiagType.NonUnit,
            UpLo.Dense, TransType.NoTrans,
            A, B);

    /// <summary>
    /// B += alpha * Trans?(A)
    /// </summary>
    public static void Axpy
        (DiagType aDiag, UpLo aUplo, TransType aTrans,
        scalar alpha, in matrix A, in matrix B)
    {
        var (m, n) = CheckLength(A, aTrans, B);
        if (m == 0 || n == 0) return;

        var AEffective = A;
        if (aTrans.HasFlag(TransType.OnlyTrans))
        {
            aUplo = Transpose(aUplo);
            AEffective = A.T;
        }
        var BEffective = B;
        var invoker = UFunc.OrDefault<MultiplyAddOperator<scalar>>(null);
        if (B.RowStride < B.ColStride)
        {
            AEffective = AEffective.T;
            BEffective = BEffective.T;
            aUplo = Transpose(aUplo);
        }
        (m, n) = GetLengths(AEffective);

        if (aUplo is UpLo.Dense)
        {
            AEffective.Combine<MultiplyAddOperator<scalar>, scalar>(alpha, BEffective, invoker);
        }
        else if (aUplo is UpLo.Lower or UpLo.Upper)
        {
            if (aUplo is UpLo.Upper)
            {
                for (int i = 0; i < m; i++)
                {
                    var start = A.DiagOffset + i;
                    if (aDiag is DiagType.Unit)
                        start++;
                    start = Math.Max(start, 0);
                    if (start >= n)
                        break;
                    var rowA = AEffective.SliceRowUncheck(i, start);
                    var rowB = BEffective.SliceRowUncheck(i, start);
                    rowA.Combine<MultiplyAddOperator<scalar>, scalar>(alpha, rowB, invoker);
                }
            }
            else // if (aUplo is UpLo.Lower)
            {
                for (int i = 0; i < m; i++)
                {
                    var end = A.DiagOffset + i + 1;
                    if (aDiag is DiagType.Unit)
                        end--;
                    end = Math.Min(end, n);
                    if (end <= 0)
                        continue;
                    var rowA = AEffective.SliceRowUncheck(i, 0, end);
                    var rowB = BEffective.SliceRowUncheck(i, 0, end);
                    rowA.Combine<MultiplyAddOperator<scalar>, scalar>(alpha, rowB, invoker);
                }
            }
        }
    }

    /// <summary>
    /// B += alpha * Trans?(A)
    /// </summary>
    public static void Axpy(scalar alpha,
        in matrix A, in matrix B)
        => Axpy(DiagType.NonUnit,
            UpLo.Dense, TransType.NoTrans,
            alpha, A, B);

    /// <summary>
    /// B = Trans?(A)
    /// </summary>
    public static void Copy
        (DiagType aDiag, UpLo aUplo, TransType aTrans, int diagOffset,
        in matrix A, in matrix B)
    {
        var (m, n) = CheckLength(A, aTrans, B);
        if (m == 0 || n == 0) return;

        var AEffective = A;
        if (aTrans.HasFlag(TransType.OnlyTrans))
        {
            aUplo = Transpose(aUplo);
            AEffective = A.T;
        }
        var BEffective = B;
        var invoker = UFunc.OrDefault<IdentityOperator<scalar>>(null);
        if (B.RowStride < B.ColStride)
        {
            AEffective = AEffective.T;
            BEffective = BEffective.T;
            aUplo = Transpose(aUplo);
        }
        (m, n) = GetLengths(AEffective);

        if (aUplo is UpLo.Dense)
        {
            AEffective.Map<IdentityOperator<scalar>>(BEffective, invoker);
        }
        else if (aUplo is UpLo.Lower or UpLo.Upper)
        {
            if (aUplo is UpLo.Upper)
            {
                for (int i = 0; i < m; i++)
                {
                    var start = A.DiagOffset + i;
                    if (aDiag is DiagType.Unit)
                        start++;
                    start = Math.Max(start, 0);
                    if (start >= n)
                        break;
                    var rowA = AEffective.SliceRowUncheck(i, start);
                    var rowB = BEffective.SliceRowUncheck(i, start);
                    rowA.Map<IdentityOperator<scalar>>(rowB, invoker);
                }
            }
            else // if (aUplo is UpLo.Lower)
            {
                for (int i = 0; i < m; i++)
                {
                    var end = A.DiagOffset + i + 1;
                    if (aDiag is DiagType.Unit)
                        end--;
                    end = Math.Min(end, n);
                    if (end <= 0)
                        continue;
                    var rowA = AEffective.SliceRowUncheck(i, 0, end);
                    var rowB = BEffective.SliceRowUncheck(i, 0, end);
                    rowA.Map<IdentityOperator<scalar>>(rowB, invoker);
                }
            }
        }
    }

    /// <summary>
    /// B = Trans?(A)
    /// </summary>
    public static void Copy(in matrix A, in matrix B)
        => Copy(DiagType.NonUnit,
            UpLo.Dense, TransType.NoTrans, 0,
            A, B);

    /// <summary>
    /// A = A / alpha
    /// </summary>
    public static void InvScal(UpLo aUplo, scalar alpha, in matrix A)
    {
        var (m, n) = CheckUploMatLength(A, aUplo);
        ArgumentOutOfRangeException.ThrowIfEqual(alpha, 0, nameof(alpha));
        if (m == 0 || n == 0) return;
        Scal(aUplo, 1 / alpha, A);
    }

    /// <summary>
    /// A = A / alpha
    /// </summary>
    public static void InvScal(scalar alpha,
        in matrix A)
        => InvScal(UpLo.Dense, alpha, A);

    /// <summary>
    /// A = alpha * A
    /// </summary>
    public static void Scal(UpLo aUplo, scalar alpha, in matrix A)
    {
        var (m, n) = CheckUploMatLength(A, aUplo);
        if (m == 0 || n == 0) return;

        var AEffective = A;
        var invoker = UFunc.OrDefault<MultiplyOperator<scalar>>(null);
        if (A.RowStride < A.ColStride)
        {
            aUplo = Transpose(aUplo);
            AEffective = A.T;
        }
        (m, n) = GetLengths(AEffective);

        if (alpha == 0.0)
        {
            Set(DiagType.NonUnit, aUplo, 0.0, A);
            return;
        }

        if (aUplo is UpLo.Dense)
        {
            AEffective.Map<MultiplyOperator<scalar>, scalar>(alpha, invoker);
        }
        else if (aUplo is UpLo.Upper)
        {
            for (int i = 0; i < m; i++)
            {
                var start = Math.Max(A.DiagOffset + i, 0);
                if (start >= n)
                    break;
                var rowA = AEffective.SliceRowUncheck(i, start);
                rowA.Map<MultiplyOperator<scalar>, scalar>(alpha, invoker);
            }
        }
        else if (aUplo is UpLo.Lower)
        {
            for (int i = 0; i < m; i++)
            {
                var end = Math.Min(n, A.DiagOffset + i + 1);
                if (end <= 0)
                    continue;
                var rowA = AEffective.SliceRowUncheck(i, 0, end);
                rowA.Map<MultiplyOperator<scalar>, scalar>(alpha, invoker);
            }
        }
        else
        {
            //A.Diag.Map<MultiplyOperator<scalar>, scalar>(alpha, invoker);
        }
    }

    /// <summary>
    /// A = alpha * A
    /// </summary>
    public static void Scal(scalar alpha, in matrix A)
        => Scal(UpLo.Dense, alpha, A);

    /// <summary>
    /// B = alpha * Trans?(A)
    /// </summary>
    public static void Scal2
        (DiagType aDiag, UpLo aUplo, TransType aTrans,
        scalar alpha, in matrix A, in matrix B)
    {
        var (m, n) = CheckLength(A, aTrans, B);
        if (m == 0 || n == 0) return;


        var AEffective = A;
        if (aTrans.HasFlag(TransType.OnlyTrans))
        {
            aUplo = Transpose(aUplo);
            AEffective = A.T;
        }
        var BEffective = B;
        var invoker = UFunc.OrDefault<MultiplyOperator<scalar>>(null);
        if (B.RowStride < B.ColStride)
        {
            AEffective = AEffective.T;
            BEffective = BEffective.T;
            aUplo = Transpose(aUplo);
        }
        (m, n) = GetLengths(AEffective);

        if (aUplo is UpLo.Dense)
        {
            AEffective.Map<MultiplyOperator<scalar>, scalar>(alpha, BEffective, invoker);
        }
        else if (aUplo is UpLo.Lower or UpLo.Upper)
        {
            if (aUplo is UpLo.Upper)
            {
                for (int i = 0; i < m; i++)
                {
                    var start = A.DiagOffset + i;
                    if (aDiag is DiagType.Unit)
                        start++;
                    start = Math.Max(start, 0);
                    if (start >= n)
                        break;
                    var rowA = AEffective.SliceRowUncheck(i, start);
                    var rowB = BEffective.SliceRowUncheck(i, start);
                    rowA.Map<MultiplyOperator<scalar>, scalar>(alpha, rowB, invoker);
                }
            }
            else // if (aUplo is UpLo.Lower)
            {
                for (int i = 0; i < m; i++)
                {
                    var end = A.DiagOffset + i + 1;
                    if (aDiag is DiagType.Unit)
                        end--;
                    end = Math.Min(end, n);
                    if (end <= 0)
                        continue;
                    var rowA = AEffective.SliceRowUncheck(i, 0, end);
                    var rowB = BEffective.SliceRowUncheck(i, 0, end);
                    rowA.Map<MultiplyOperator<scalar>, scalar>(alpha, rowB, invoker);
                }
            }
        }
    }

    /// <summary>
    /// B = alpha * Trans?(A)
    /// </summary>
    public static void Scal2(scalar alpha, in matrix A, in matrix B)
        => Scal2(DiagType.NonUnit,
            UpLo.Dense, TransType.NoTrans,
            alpha, A, B);

    /// <summary>
    /// A = alpha
    /// </summary>
    public static void Set
        (DiagType aDiag, UpLo aUplo, scalar alpha, in matrix A)
        => Set(aDiag, aUplo, alpha, A, A.DiagOffset);

    /// <summary>
    /// A = alpha
    /// </summary>
    public static void Set
        (DiagType aDiag, UpLo aUplo, scalar alpha, in matrix A, int diag)
    {
        var (m, n) = GetLengths(A);
        if (m == 0 || n == 0) return;


        var AEffective = A;
        var invoker = UFunc.OrDefault<IdentityOperator<scalar>>(null);
        if (A.RowStride < A.ColStride)
        {
            aUplo = Transpose(aUplo);
            AEffective = A.T;
        }
        (m, n) = GetLengths(AEffective);

        if (aUplo is UpLo.Dense)
        {
            AEffective.Apply<IdentityOperator<scalar>, scalar>(alpha, invoker);
        }
        else if (aUplo is UpLo.Upper)
        {
            for (int i = 0; i < m; i++)
            {
                var start = Math.Max(A.DiagOffset + i, 0);
                if (aDiag is DiagType.Unit)
                {
                    start++;
                }
                if (start >= n)
                    break;
                var rowA = AEffective.SliceRowUncheck(i, start);
                rowA.Apply<IdentityOperator<scalar>, scalar>(alpha, invoker);
            }
        }
        else if (aUplo is UpLo.Lower)
        {
            for (int i = 0; i < m; i++)
            {
                var end = Math.Min(n, A.DiagOffset + i + 1);
                if (aDiag is DiagType.Unit)
                {
                    end--;
                }
                if (end <= 0)
                    continue;
                var rowA = AEffective.SliceRowUncheck(i, 0, end);
                rowA.Apply<IdentityOperator<scalar>, scalar>(alpha, invoker);
            }
        }
    }

    /// <summary>
    /// A = alpha
    /// </summary>
    public static void Set(scalar alpha,
        in matrix A)
        => Set(DiagType.NonUnit, UpLo.Dense,
            alpha, A, A.DiagOffset);

    /// <summary>
    /// B -= Trans?(A)
    /// </summary>
    public static void Sub
        (DiagType aDiag, UpLo aUplo, TransType aTrans,
        in matrix A, in matrix B)
    {
        var (m, n) = CheckLength(A, aTrans, B);
        if (m == 0 || n == 0) return;


        var AEffective = A;
        if (aTrans.HasFlag(TransType.OnlyTrans))
        {
            aUplo = Transpose(aUplo);
            AEffective = A.T;
        }
        var BEffective = B;
        var invoker = UFunc.OrDefault<ReversedOp<SubtractOperator<scalar>, scalar, scalar, scalar>>(null);
        if (B.RowStride < B.ColStride)
        {
            AEffective = AEffective.T;
            BEffective = BEffective.T;
            aUplo = Transpose(aUplo);
        }
        (m, n) = GetLengths(AEffective);

        if (aUplo is UpLo.Dense)
        {
            AEffective.Combine<ReversedOp<SubtractOperator<scalar>, scalar, scalar, scalar>>(BEffective, invoker);
        }
        else if (aUplo is UpLo.Lower or UpLo.Upper)
        {
            if (aUplo is UpLo.Upper)
            {
                for (int i = 0; i < m; i++)
                {
                    var start = A.DiagOffset + i;
                    if (aDiag is DiagType.Unit)
                        start++;
                    start = Math.Max(start, 0);
                    if (start >= n)
                        break;
                    var rowA = AEffective.SliceRowUncheck(i, start);
                    var rowB = BEffective.SliceRowUncheck(i, start);
                    rowA.Combine<ReversedOp<SubtractOperator<scalar>, scalar, scalar, scalar>>(rowB, invoker);
                }
            }
            else // if (aUplo is UpLo.Lower)
            {
                for (int i = 0; i < m; i++)
                {
                    var end = A.DiagOffset + i + 1;
                    if (aDiag is DiagType.Unit)
                        end--;
                    end = Math.Min(end, n);
                    if (end <= 0)
                        continue;
                    var rowA = AEffective.SliceRowUncheck(i, 0, end);
                    var rowB = BEffective.SliceRowUncheck(i, 0, end);
                    rowA.Combine<ReversedOp<SubtractOperator<scalar>, scalar, scalar, scalar>>(rowB, invoker);
                }
            }
        }
    }

    /// <summary>
    /// B -= Trans?(A)
    /// </summary>
    public static void Sub(in matrix A, in matrix B)
        => Sub(DiagType.NonUnit,
            UpLo.Dense, TransType.NoTrans,
            A, B);
}
