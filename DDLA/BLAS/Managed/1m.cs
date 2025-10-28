using DDLA.UFuncs.Operators;
using DDLA.Misc.Flags;

using scalar = double;
using vector = DDLA.Core.VectorView;
using matrix = DDLA.Core.MatrixView;
using DDLA.UFuncs;
using System.Runtime.CompilerServices;

namespace DDLA.BLAS.Managed;

public static partial class BlasProvider
{
    public static void Add(matrix A, matrix B)
        => A.Combine<AddOperator<scalar>>(B);

    public static void Axpy(scalar alpha, matrix A, matrix B)
        => A.Combine<MultiplyAddOperator<scalar>, scalar>(alpha, B);

    public static void Copy(matrix A, matrix B)
        => A.Map<IdentityOperator<scalar>>(B);

    public static void InvScal(scalar alpha, matrix A)
    {
        if (alpha == 0.0 || alpha == -0.0 || !scalar.IsFinite(alpha))
        {
            throw new ArgumentException("Error: alpha must be A finite non-zero value.");
        }
        A.Map<MultiplyOperator<scalar>, scalar>(1 / alpha);
    }

    public static void Scal(scalar alpha, matrix A)
    {
        if (alpha is 0 or -0)
            A.Apply<IdentityOperator<scalar>, scalar>(0);
        else if(alpha is -1)
            A.Map<NegateOperator<scalar>>();
        else if (alpha is not 1)
            A.Map<MultiplyOperator<scalar>, scalar>(alpha);
    }

    public static void Scal2(scalar alpha, matrix A, 
        matrix B)
    {
        if (alpha is 0 or -0)
            B.Apply<IdentityOperator<scalar>, scalar>(0);
        else
            A.Map<MultiplyOperator<scalar>, scalar>(alpha, B);
    }

    public static void Set(scalar alpha, matrix A)
        => A.Apply<IdentityOperator<scalar>, scalar>(alpha);

    public static void Sub(matrix A, matrix B)
        => A.Combine<SubtractOperator<scalar>>(B);

    /// <summary>
    /// B += Trans?(A)
    /// </summary>
    public static void Add(DiagType aDiag, UpLo aUplo, TransType aTrans,
        in matrix A, in matrix B)
    {
        var (m, n) = CheckLength(A, aTrans, B);
        if (m == 0 || n == 0) return;
        Source.Add(A.DiagOffset, aDiag, aUplo, aTrans,
            m, n,
            ref A.GetHeadRef(), A.RowStride, A.ColStride,
            ref B.GetHeadRef(), B.RowStride, B.ColStride);
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
        Source.Axpy(A.DiagOffset, aDiag, aUplo, aTrans,
            m, n,
            alpha,
            ref A.GetHeadRef(), A.RowStride, A.ColStride,
            ref B.GetHeadRef(), B.RowStride, B.ColStride);
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
        Source.Copy(A.DiagOffset, aDiag, aUplo, aTrans,
            m, n,
            ref A.GetHeadRef(), A.RowStride, A.ColStride,
            ref B.GetHeadRef(), B.RowStride, B.ColStride);
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
        ref var head = ref A.GetHeadRef();

        if (alpha == 1) return;
        else if (aUplo == UpLo.Dense)
        {
            if (alpha is 0 or -0)
                A.Apply<IdentityOperator<scalar>, scalar>(0);
            else if (alpha is -1)
                A.Map<NegateOperator<scalar>>();
            else
                A.Map<MultiplyOperator<scalar>, scalar>(alpha);
        }
        else
        {
            var AEffective = A;
            if (A.ColStride > A.RowStride)
            {
                AEffective = A.T;
                aUplo = Transpose(aUplo);
                (m, n) = (n, m);
            }

            if (aUplo is UpLo.Upper)
            {
                m = Math.Min(m, n);
                for(int i = 0; i < m; i++)
                {
                    var row = AEffective.SliceRowUncheck(i, i);
                    row.Map<MultiplyOperator<double>, double>(alpha);
                }
            }
            else if (aUplo is UpLo.Lower)
            {
                for (int i = 0; i < m; i++)
                {
                    var row = AEffective.SliceRowUncheck(i, 0, Math.Min(i + 1, n));
                    row.Map<MultiplyOperator<double>, double>(alpha);
                }
            }
            else
            {
                A.Diag.Map<MultiplyOperator<double>, double>(alpha);
            }
        }
    }

    /// <summary>
    /// A = alpha * A
    /// </summary>
    public static void Scal(scalar alpha,
        in matrix A)
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
        Source.Scal2(A.DiagOffset, aDiag, aUplo, aTrans,
            m, n,
            alpha,
            ref A.GetHeadRef(), A.RowStride, A.ColStride,
            ref B.GetHeadRef(), B.RowStride, B.ColStride);
    }

    /// <summary>
    /// B = alpha * Trans?(A)
    /// </summary>
    public static void Scal2(double alpha, in matrix A, in matrix B)
        => Scal2(DiagType.NonUnit,
            UpLo.Dense, TransType.NoTrans,
            alpha, A, B);

    /// <summary>
    /// B = alpha
    /// </summary>
    public static void Set
        (DiagType aDiag, UpLo aUplo, scalar alpha, in matrix A)
    {
        var (m, n) = CheckUploMatLength(A, aUplo);
        if (m == 0 || n == 0) return;
        Source.Set(ConjType.NoConj, A.DiagOffset, aDiag, aUplo,
            m, n,
            alpha,
            ref A.GetHeadRef(), A.RowStride, A.ColStride);
    }

    /// <summary>
    /// B = alpha
    /// </summary>
    public static void Set
        (DiagType aDiag, UpLo aUplo, scalar alpha, in matrix A, int diag)
    {
        var (m, n) = CheckUploMatLength(A, aUplo);
        if (m == 0 || n == 0) return;
        Source.Set(ConjType.NoConj, diag, aDiag, aUplo,
            m, n,
            alpha,
            ref A.GetHeadRef(), A.RowStride, A.ColStride);
    }

    /// <summary>
    /// B = alpha
    /// </summary>
    public static void Set(scalar alpha,
        in matrix A)
        => Set(DiagType.NonUnit, UpLo.Dense,
            alpha, A);

    /// <summary>
    /// B -= Trans?(A)
    /// </summary>
    public static void Sub
        (DiagType aDiag, UpLo aUplo, TransType aTrans,
        in matrix A, in matrix B)
    {
        var (m, n) = CheckLength(A, aTrans, B);
        if (m == 0 || n == 0) return;
        Source.Sub(A.DiagOffset, aDiag, aUplo, aTrans,
            m, n,
            ref A.GetHeadRef(), A.RowStride, A.ColStride,
            ref B.GetHeadRef(), B.RowStride, B.ColStride);
    }

    /// <summary>
    /// B -= Trans?(A)
    /// </summary>
    public static void Sub(in matrix A, in matrix B)
        => Sub(DiagType.NonUnit,
            UpLo.Dense, TransType.NoTrans,
            A, B);

}
