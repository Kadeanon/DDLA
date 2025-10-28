using DDLA.UFuncs.Operators;
using DDLA.Misc.Flags;

using scalar = double;
using vector = DDLA.Core.VectorView;
using matrix = DDLA.Core.MatrixView;
using DDLA.UFuncs;

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
        Details.Combine<AddOperator<scalar>>(aDiag, aUplo, aTrans, A, B);
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
        Details.Combine<MultiplyAddOperator<scalar>, scalar>(aDiag, aUplo, aTrans,
            alpha, A, B);
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
        (DiagType aDiag, UpLo aUplo, TransType aTrans,
        in matrix A, in matrix B)
    {
        var (m, n) = CheckLength(A, aTrans, B);
        if (m == 0 || n == 0) return;
        Details.Map<IdentityOperator<scalar>>(aDiag, aUplo, aTrans, A, B);
    }

    /// <summary>
    /// B = Trans?(A)
    /// </summary>
    public static void Copy(in matrix A, in matrix B)
        => Copy(DiagType.NonUnit,
            UpLo.Dense, TransType.NoTrans,
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
        Details.Map<MultiplyOperator<scalar>, scalar>(aUplo, alpha, A);
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
        Details.Map<MultiplyOperator<scalar>, scalar>(aDiag, aUplo, aTrans,
            alpha, A, B);
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
        Details.Set_Kernel(aDiag, aUplo, alpha, A, diag);
    }

    /// <summary>
    /// A = alpha
    /// </summary>
    public static void Set(scalar alpha,
        in matrix A)
        => Set(DiagType.NonUnit, UpLo.Dense,
            alpha, A, A.DiagOffset);

    public static partial class Details
    {
        public static void Set_Kernel(DiagType aDiag, UpLo aUplo,
            scalar alpha, in matrix A, int diag)
        {
            var AEffective = A;
            var invoker = UFunc.OrDefault<IdentityOperator<scalar>>(null);
            if (A.RowStride < A.ColStride)
            {
                aUplo = Transpose(aUplo);
                AEffective = A.T;
            }
            var (m, n) = GetLengths(AEffective);

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
    }

    /// <summary>
    /// B -= Trans?(A)
    /// </summary>
    public static void Sub
        (DiagType aDiag, UpLo aUplo, TransType aTrans,
        in matrix A, in matrix B)
    {
        var (m, n) = CheckLength(A, aTrans, B);
        if (m == 0 || n == 0) return;
        Details.Combine<ReversedOp<SubtractOperator<scalar>, scalar, scalar, scalar>>(aDiag, aUplo, aTrans, A, B);
    }

    /// <summary>
    /// B -= Trans?(A)
    /// </summary>
    public static void Sub(in matrix A, in matrix B)
        => Sub(DiagType.NonUnit,
            UpLo.Dense, TransType.NoTrans,
            A, B);

    public static partial class Details
    {
        public static void Map<TAction, TScalar>(UpLo aUplo,
             TScalar alpha, in matrix A)
            where TAction: struct, IBinaryOperator<scalar, TScalar, scalar>
            where TScalar: struct
        {
            var AEffective = A;
            var invoker = UFunc.OrDefault<TAction>(null);
            if (A.RowStride < A.ColStride)
            {
                aUplo = Transpose(aUplo);
                AEffective = A.T;
            }
            var (m, n) = GetLengths(AEffective);

            if (aUplo is UpLo.Dense)
            {
                AEffective.Map<TAction, TScalar>(alpha, invoker);
            }
            else if (aUplo is UpLo.Upper)
            {
                for (int i = 0; i < m; i++)
                {
                    var start = Math.Max(A.DiagOffset + i, 0);
                    if (start >= n)
                        break;
                    var rowA = AEffective.SliceRowUncheck(i, start);
                    rowA.Map<TAction, TScalar>(alpha, invoker);
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
                    rowA.Map<TAction, TScalar>(alpha, invoker);
                }
            }
            else
            {
                A.Diag.Map<TAction, TScalar>(alpha, invoker);
            }
        }

        public static void Map<TAction>(DiagType aDiag, UpLo aUplo, TransType aTrans,
            in matrix A, in matrix B)
            where TAction : struct, IUnaryOperator<scalar, scalar>
        {
            var AEffective = A;
            var BEffective = B;
            var invoker = UFunc.OrDefault<TAction>(null);
            if (B.RowStride < B.ColStride)
            {
                AEffective = A.T;
                BEffective = B.T;
                aUplo = Transpose(aUplo);
                aTrans = Transpose(aTrans);
            }
            if (aTrans.HasFlag(TransType.OnlyTrans))
            {
                aUplo = Transpose(aUplo);
                AEffective = A.T;
            }
            var (m, n) = GetLengths(AEffective);

            if (aUplo is UpLo.Dense)
            {
                AEffective.Map<TAction>(BEffective, invoker);
            }
            else if (aUplo is UpLo.Upper)
            {
                for (int i = 0; i < m; i++)
                {
                    var start = Math.Max(A.DiagOffset + i, 0);
                    if (aDiag is DiagType.Unit)
                    {
                        ref var diagRefB = ref BEffective.AtUncheck(i, start);
                        diagRefB = invoker.Invoke(1.0);
                        start++;
                    }
                    if (start >= n)
                        break;
                    var rowA = AEffective.SliceRowUncheck(i, start);
                    var rowB = BEffective.SliceRowUncheck(i, start);
                    rowA.Map<TAction>(rowB, invoker);
                }
            }
            else if (aUplo is UpLo.Lower)
            {
                for (int i = 0; i < m; i++)
                {
                    var end = Math.Min(n, A.DiagOffset + i + 1);
                    if (aDiag is DiagType.Unit)
                    {
                        ref var diagRefB = ref BEffective.AtUncheck(i, end - 1);
                        diagRefB = invoker.Invoke(1.0);
                        end--;
                    }
                    if (end <= 0)
                        continue;
                    var rowA = AEffective.SliceRowUncheck(i, 0, end);
                    var rowB = BEffective.SliceRowUncheck(i, 0, end);
                    rowA.Map<TAction>(rowB, invoker);
                }
            }
            else
            {
                if (aDiag is DiagType.Unit)
                {
                    var diag = B.Diag;
                    for (int i = 0; i < diag.Length; i++)
                    {
                        ref var diagRefB = ref diag.AtUncheck(i);
                        diagRefB = invoker.Invoke(1.0);
                    }
                }
                else
                {
                    A.Diag.Map<TAction>(B.Diag, invoker);
                }
            }
        }

        public static void Map<TAction, TScalar>(DiagType aDiag, UpLo aUplo, TransType aTrans,
            TScalar alpha, in matrix A, in matrix B)
            where TAction : struct, IBinaryOperator<scalar, TScalar, scalar>
            where TScalar : struct
        {
            var AEffective = A;
            var BEffective = B;
            var invoker = UFunc.OrDefault<TAction>(null);
            if (B.RowStride < B.ColStride)
            {
                AEffective = A.T;
                BEffective = B.T;
                aUplo = Transpose(aUplo);
                aTrans = Transpose(aTrans);
            }
            if (aTrans.HasFlag(TransType.OnlyTrans))
            {
                aUplo = Transpose(aUplo);
                AEffective = A.T;
            }
            var (m, n) = GetLengths(AEffective);

            if (aUplo is UpLo.Dense)
            {
                AEffective.Map<TAction, TScalar>(alpha, BEffective, invoker);
            }
            else if (aUplo is UpLo.Upper)
            {
                for (int i = 0; i < m; i++)
                {
                    var start = Math.Max(A.DiagOffset + i, 0);
                    if (aDiag is DiagType.Unit)
                    {
                        ref var diagRefB = ref BEffective.AtUncheck(i, start);
                        diagRefB = invoker.Invoke(1.0, alpha);
                        start++;
                    }
                    if (start >= n)
                        break;
                    var rowA = AEffective.SliceRowUncheck(i, start);
                    var rowB = BEffective.SliceRowUncheck(i, start);
                    rowA.Map<TAction, TScalar>(alpha, rowB, invoker);
                }
            }
            else if (aUplo is UpLo.Lower)
            {
                for (int i = 0; i < m; i++)
                {
                    var end = Math.Min(n, A.DiagOffset + i + 1);
                    if (aDiag is DiagType.Unit)
                    {
                        ref var diagRefB = ref BEffective.AtUncheck(i, end - 1);
                        diagRefB = invoker.Invoke(1.0, alpha);
                        end--;
                    }
                    if (end <= 0)
                        continue;
                    var rowA = AEffective.SliceRowUncheck(i, 0, end);
                    var rowB = BEffective.SliceRowUncheck(i, 0, end);
                    rowA.Map<TAction, TScalar>(alpha, rowB, invoker);
                }
            }
            else
            {
                if (aDiag is DiagType.Unit)
                {
                    var diag = B.Diag;
                    for (int i = 0; i < diag.Length; i++)
                    {
                        ref var diagRefB = ref diag.AtUncheck(i);
                        diagRefB = invoker.Invoke(1.0, alpha);
                    }
                }
                else
                {
                    A.Diag.Map<TAction, TScalar>(alpha, B.Diag, invoker);
                }
            }
        }

        public static void Combine<TAction>(DiagType aDiag, UpLo aUplo, TransType aTrans,
            in matrix A, in matrix B)
            where TAction : struct, IBinaryOperator<scalar, scalar, scalar>
        {
            var AEffective = A;
            var BEffective = B;
            var invoker = UFunc.OrDefault<TAction>(null);
            if (B.RowStride < B.ColStride)
            {
                AEffective = A.T;
                BEffective = B.T;
                aUplo = Transpose(aUplo);
                aTrans = Transpose(aTrans);
            }
            if (aTrans.HasFlag(TransType.OnlyTrans))
            {
                aUplo = Transpose(aUplo);
                AEffective = A.T;
            }
            var (m, n) = GetLengths(AEffective);

            if (aUplo is UpLo.Dense)
            {
                AEffective.Combine<TAction>(BEffective, invoker);
            }
            else if (aUplo is UpLo.Upper)
            {
                for (int i = 0; i < m; i++)
                {
                    var start = Math.Max(A.DiagOffset + i, 0);
                    if (aDiag is DiagType.Unit)
                    {
                        ref var diagRefB = ref BEffective.AtUncheck(i, start);
                        diagRefB = invoker.Invoke(1.0, diagRefB);
                        start++;
                    }
                    if (start >= n)
                        break;
                    var rowA = AEffective.SliceRowUncheck(i, start);
                    var rowB = BEffective.SliceRowUncheck(i, start);
                    rowA.Combine<TAction>(rowB, invoker);
                }
            }
            else if (aUplo is UpLo.Lower)
            {
                for (int i = 0; i < m; i++)
                {
                    var end = Math.Min(n, A.DiagOffset + i + 1);
                    if (aDiag is DiagType.Unit)
                    {
                        ref var diagRefB = ref BEffective.AtUncheck(i, end - 1);
                        diagRefB = invoker.Invoke(0.0, diagRefB);
                        end--;
                    }
                    if (end <= 0)
                        continue;
                    var rowA = AEffective.SliceRowUncheck(i, 0, end);
                    var rowB = BEffective.SliceRowUncheck(i, 0, end);
                    rowA.Combine<TAction>(rowB, invoker);
                }
            }
            else
            {
                if (aDiag is DiagType.Unit)
                {
                    var diag = B.Diag;
                    for (int i = 0; i < diag.Length; i++)
                    {
                        ref var diagRefB = ref diag.AtUncheck(i);
                        diagRefB = invoker.Invoke(1.0, diagRefB);
                    }
                }
                else
                {
                    A.Diag.Combine<TAction>(B.Diag, invoker);
                }
            }
        }

        public static void Combine<TAction, TScalar>(DiagType aDiag, UpLo aUplo, TransType aTrans,
            TScalar alpha, in matrix A, in matrix B)
                where TAction : struct, ITernaryOperator<scalar, TScalar, scalar, scalar>
                where TScalar : struct
        {
            var AEffective = A;
            var BEffective = B;
            var invoker = UFunc.OrDefault<TAction>(null);
            if (B.RowStride < B.ColStride)
            {
                AEffective = A.T;
                BEffective = B.T;
                aUplo = Transpose(aUplo);
                aTrans = Transpose(aTrans);
            }
            if (aTrans.HasFlag(TransType.OnlyTrans))
            {
                aUplo = Transpose(aUplo);
                AEffective = A.T;
            }
            var (m, n) = GetLengths(AEffective);


            if (aUplo is UpLo.Dense)
            {
                AEffective.Combine<TAction, TScalar>(alpha, BEffective, invoker);
            }
            else if (aUplo is UpLo.Upper)
            {
                for (int i = 0; i < m; i++)
                {
                    var start = Math.Max(A.DiagOffset + i, 0);
                    if (aDiag is DiagType.Unit)
                    {
                        ref var diagRefB = ref BEffective.AtUncheck(i, start);
                        diagRefB = invoker.Invoke(1.0, alpha, diagRefB);
                        start++;
                    }
                    if (start >= n)
                        break;
                    var rowA = AEffective.SliceRowUncheck(i, start);
                    var rowB = BEffective.SliceRowUncheck(i, start);
                    rowA.Combine<TAction, TScalar>(alpha, rowB, invoker);
                }
            }
            else if (aUplo is UpLo.Lower)
            {
                for (int i = 0; i < m; i++)
                {
                    var end = Math.Min(n, A.DiagOffset + i + 1);
                    if (aDiag is DiagType.Unit)
                    {
                        ref var diagRefB = ref BEffective.AtUncheck(i, end - 1);
                        diagRefB = invoker.Invoke(1.0, alpha, diagRefB);
                        end--;
                    }
                    if (end <= 0)
                        continue;
                    var rowA = AEffective.SliceRowUncheck(i, 0, end);
                    var rowB = BEffective.SliceRowUncheck(i, 0, end);
                    rowA.Combine<TAction, TScalar>(alpha, rowB, invoker);
                }
            }
            else
            {
                if (aDiag is DiagType.Unit)
                {
                    var diag = B.Diag;
                    for (int i = 0; i < diag.Length; i++)
                    {
                        ref var diagRefB = ref diag.AtUncheck(i);
                        diagRefB = invoker.Invoke(1.0, alpha, diagRefB);
                    }
                }
                else
                {
                    A.Diag.Combine<TAction, TScalar>(alpha, B.Diag, invoker);
                }
            }
        }
    }
}
