using DDLA.Misc.Flags;
using DDLA.UFuncs;
using DDLA.UFuncs.Operators;
using matrix = DDLA.Core.MatrixView;
using scalar = double;

namespace DDLA.BLAS.Managed;

public static partial class BlasProvider
{
    /// <summary>
    /// Performs an element-wise matrix addition.<br />
    /// <paramref name="B"/> := <paramref name="B"/> + 
    /// Trans?(Uplo?(<paramref name="A"/>)), <br />
    /// using <paramref name="aDiag"/> and <paramref name="aUplo"/> to specify
    /// the region on which the addition is performed, and
    /// <paramref name="aTrans"/> to specify whether a transposed view of
    /// <paramref name="A"/> is used.
    /// </summary>
    /// <param name="aDiag">
    /// When <paramref name="aUplo"/> is not <see cref="UpLo.Dense"/>,
    /// specifies whether the diagonal elements are included in the addition:<br />
    /// - <see cref="DiagType.NonUnit"/>: operate on diagonal elements as usual;<br />
    /// - <see cref="DiagType.Unit"/>: skip diagonal elements in the addition.
    /// </param>
    /// <param name="aUplo">
    /// Specifies the region of the matrix to be processed:<br />
    /// - <see cref="UpLo.Dense"/>: the entire matrix;<br />
    /// - <see cref="UpLo.Upper"/>: only the upper triangular part;<br />
    /// - <see cref="UpLo.Lower"/>: only the lower triangular part.
    /// </param>
    /// <param name="aTrans">
    /// Specifies whether <paramref name="A"/> is transposed:<br />
    /// - <see cref="TransType.NoTrans"/> or 
    ///   <see cref="TransType.OnlyConj"/>: no transpose;<br />
    /// - <see cref="TransType.OnlyTrans"/> or 
    ///   <see cref="TransType.ConjTrans"/>: transpose.
    /// </param>
    /// <param name="A">Input matrix A.</param>
    /// <param name="B">Input/output matrix B.</param>
    /// <remarks>
    /// If <paramref name="aTrans"/> requests a transpose,
    /// the transposed matrix <paramref name="A"/> must be conformable
    /// with <paramref name="B"/>.
    /// </remarks>
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
                    var start = i;
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
                    var end = i + 1;
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
    /// Performs an element-wise matrix addition.<br />
    /// <paramref name="B"/> := <paramref name="B"/> + <paramref name="A"/>.
    /// </summary>
    /// <param name="A">Input matrix A.</param>
    /// <param name="B">Input/output matrix B.</param>
    public static void Add(in matrix A, in matrix B)
        => Add(DiagType.NonUnit,
            UpLo.Dense, TransType.NoTrans,
            A, B);

    /// <summary>
    /// Performs an element-wise scaled matrix addition.<br />
    /// <paramref name="B"/> := <paramref name="B"/> + 
    /// <paramref name="alpha"/> * Trans?(Uplo?(<paramref name="A"/>)), <br />
    /// using <paramref name="aDiag"/> and <paramref name="aUplo"/> to specify
    /// the region on which the addition is performed, and
    /// <paramref name="aTrans"/> to specify whether a transposed view of
    /// <paramref name="A"/> is used.
    /// </summary>
    /// <param name="aDiag">
    /// When <paramref name="aUplo"/> is not <see cref="UpLo.Dense"/>,
    /// specifies whether the diagonal elements are included in the addition:<br />
    /// - <see cref="DiagType.NonUnit"/>: operate on diagonal elements as usual;<br />
    /// - <see cref="DiagType.Unit"/>: skip diagonal elements in the addition.
    /// </param>
    /// <param name="aUplo">
    /// Specifies the region of the matrix to be processed:<br />
    /// - <see cref="UpLo.Dense"/>: the entire matrix;<br />
    /// - <see cref="UpLo.Upper"/>: only the upper triangular part;<br />
    /// - <see cref="UpLo.Lower"/>: only the lower triangular part.
    /// </param>
    /// <param name="aTrans">
    /// Specifies whether <paramref name="A"/> is transposed:<br />
    /// - <see cref="TransType.NoTrans"/> or 
    ///   <see cref="TransType.OnlyConj"/>: no transpose;<br />
    /// - <see cref="TransType.OnlyTrans"/> or 
    ///   <see cref="TransType.ConjTrans"/>: transpose.
    /// </param>
    /// <param name="alpha">Scaling factor applied to <paramref name="A"/>.</param>
    /// <param name="A">Input matrix A.</param>
    /// <param name="B">Input/output matrix B.</param>
    /// <remarks>
    /// If <paramref name="aTrans"/> requests a transpose,
    /// the transposed matrix <paramref name="A"/> must be conformable
    /// with <paramref name="B"/>.
    /// </remarks>
    public static void Axpy(DiagType aDiag, UpLo aUplo,
        TransType aTrans, scalar alpha, in matrix A, in matrix B)
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
                    var start = i;
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
                    var end = i + 1;
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
    /// Performs an element-wise scaled matrix addition.<br />
    /// <paramref name="B"/> := <paramref name="B"/> + 
    /// <paramref name="alpha"/> * <paramref name="A"/>.
    /// </summary>
    /// <param name="alpha">Scaling factor applied to <paramref name="A"/>.</param>
    /// <param name="A">Input matrix A.</param>
    /// <param name="B">Input/output matrix B.</param>
    public static void Axpy(scalar alpha, in matrix A,
        in matrix B)
        => Axpy(DiagType.NonUnit,
            UpLo.Dense, TransType.NoTrans,
            alpha, A, B);

    /// <summary>
    /// Performs an element-wise matrix copy.<br />
    /// <paramref name="B"/> := 
    /// Trans?(Uplo?(<paramref name="A"/>)), <br />
    /// using <paramref name="aDiag"/> and <paramref name="aUplo"/> to specify
    /// the region to be copied, and <paramref name="aTrans"/> to specify
    /// whether a transposed view of <paramref name="A"/> is used.
    /// </summary>
    /// <param name="aDiag">
    /// When <paramref name="aUplo"/> is not <see cref="UpLo.Dense"/>,
    /// specifies whether the diagonal elements are copied:<br />
    /// - <see cref="DiagType.NonUnit"/>: copy diagonal elements as usual;<br />
    /// - <see cref="DiagType.Unit"/>: skip diagonal elements when copying.
    /// </param>
    /// <param name="aUplo">
    /// Specifies the region of the matrix to be processed:<br />
    /// - <see cref="UpLo.Dense"/>: the entire matrix;<br />
    /// - <see cref="UpLo.Upper"/>: only the upper triangular part;<br />
    /// - <see cref="UpLo.Lower"/>: only the lower triangular part.
    /// </param>
    /// <param name="aTrans">
    /// Specifies whether <paramref name="A"/> is transposed:<br />
    /// - <see cref="TransType.NoTrans"/> or 
    ///   <see cref="TransType.OnlyConj"/>: no transpose;<br />
    /// - <see cref="TransType.OnlyTrans"/> or 
    ///   <see cref="TransType.ConjTrans"/>: transpose.
    /// </param>
    /// <param name="A">Input matrix A.</param>
    /// <param name="B">Output matrix B.</param>
    /// <remarks>
    /// If <paramref name="aTrans"/> requests a transpose,
    /// the transposed matrix <paramref name="A"/> must be conformable
    /// with <paramref name="B"/>.
    /// </remarks>
    public static void Copy(DiagType aDiag, UpLo aUplo,
        TransType aTrans, in matrix A, in matrix B)
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
                    var start = i;
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
                    var end = i + 1;
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
    /// Performs an element-wise matrix copy.<br />
    /// <paramref name="B"/> := <paramref name="A"/>.
    /// </summary>
    /// <param name="A">Input matrix A.</param>
    /// <param name="B">Output matrix B.</param>
    public static void Copy(in matrix A, in matrix B)
        => Copy(DiagType.NonUnit,
            UpLo.Dense, TransType.NoTrans, A, B);

    /// <summary>
    /// Performs an element-wise matrix scaling operation.<br />
    /// Uplo?(<paramref name="A"/>) := Uplo?(<paramref name="A"/>) 
    /// / <paramref name="alpha"/>, <br />
    /// using <paramref name="aUplo"/> to specify the region to be scaled.
    /// </summary>
    /// <param name="aUplo">
    /// Specifies the region of the matrix to be processed:<br />
    /// - <see cref="UpLo.Dense"/>: the entire matrix;<br />
    /// - <see cref="UpLo.Upper"/>: only the upper triangular part;<br />
    /// - <see cref="UpLo.Lower"/>: only the lower triangular part.
    /// </param>
    /// <param name="alpha">Scaling factor applied to <paramref name="A"/>.</param>
    /// <param name="A">Input/output matrix A.</param>
    /// <remarks>
    /// Since the implementation internally uses the reciprocal of
    /// <paramref name="alpha"/> to perform scaling, this may cause
    /// precision issues in extreme cases.
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when <paramref name="alpha"/> is 0, since it cannot be used
    /// as a divisor.
    /// </exception>
    public static void InvScal(UpLo aUplo, scalar alpha,
        in matrix A)
    {
        var (m, n) = CheckUploMatLength(A, aUplo);
        ArgumentOutOfRangeException.ThrowIfEqual(alpha, 0, nameof(alpha));
        if (m == 0 || n == 0) return;
        Scal(aUplo, 1 / alpha, A);
    }

    /// <summary>
    /// Performs an element-wise matrix scaling operation.<br />
    /// <paramref name="A"/> := <paramref name="A"/> 
    /// / <paramref name="alpha"/>.
    /// </summary>
    /// <param name="alpha">Scaling factor applied to <paramref name="A"/>.</param>
    /// <param name="A">Input/output matrix A.</param>
    /// <remarks>
    /// Since the implementation internally uses the reciprocal of
    /// <paramref name="alpha"/> to perform scaling, this may cause
    /// precision issues in extreme cases.
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when <paramref name="alpha"/> is 0, since it cannot be used
    /// as a divisor.
    /// </exception>
    public static void InvScal(scalar alpha, in matrix A)
        => InvScal(UpLo.Dense, alpha, A);

    /// <summary>
    /// Performs an element-wise matrix scaling operation.<br />
    /// Uplo?(<paramref name="A"/>) := <paramref name="alpha"/>
    /// * Uplo?(<paramref name="A"/>), <br />
    /// using <paramref name="aUplo"/> to specify the region to be scaled.
    /// </summary>
    /// <param name="aUplo">
    /// Specifies the region of the matrix to be processed:<br />
    /// - <see cref="UpLo.Dense"/>: the entire matrix;<br />
    /// - <see cref="UpLo.Upper"/>: only the upper triangular part;<br />
    /// - <see cref="UpLo.Lower"/>: only the lower triangular part.
    /// </param>
    /// <param name="alpha">Scaling factor applied to <paramref name="A"/>.</param>
    /// <param name="A">Input/output matrix A.</param>
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
                var start = Math.Max(i, 0);
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
                var end = Math.Min(n, i + 1);
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
    /// Performs an element-wise matrix scaling operation.<br />
    /// <paramref name="A"/> := <paramref name="alpha"/>
    /// * <paramref name="A"/>.
    /// </summary>
    /// <param name="alpha">Scaling factor applied to <paramref name="A"/>.</param>
    /// <param name="A">Input/output matrix A.</param>
    public static void Scal(scalar alpha, in matrix A)
        => Scal(UpLo.Dense, alpha, A);

    /// <summary>
    /// Performs an element-wise matrix scaling operation and stores the
    /// result in another matrix.<br />
    /// <paramref name="B"/> := <paramref name="alpha"/> 
    /// * Trans?(Uplo?(<paramref name="A"/>)), <br />
    /// using <paramref name="aDiag"/> and <paramref name="aUplo"/> to specify
    /// the region to be scaled, and <paramref name="aTrans"/> to specify
    /// whether a transposed view of <paramref name="A"/> is used.
    /// </summary>
    /// <param name="aDiag">
    /// When <paramref name="aUplo"/> is not <see cref="UpLo.Dense"/>,
    /// specifies whether the diagonal elements are scaled:<br />
    /// - <see cref="DiagType.NonUnit"/>: scale diagonal elements as usual;<br />
    /// - <see cref="DiagType.Unit"/>: skip diagonal elements when scaling.
    /// </param>
    /// <param name="aUplo">
    /// Specifies the region of the matrix to be processed:<br />
    /// - <see cref="UpLo.Dense"/>: the entire matrix;<br />
    /// - <see cref="UpLo.Upper"/>: only the upper triangular part;<br />
    /// - <see cref="UpLo.Lower"/>: only the lower triangular part.
    /// </param>
    /// <param name="aTrans">
    /// Specifies whether <paramref name="A"/> is transposed:<br />
    /// - <see cref="TransType.NoTrans"/> or 
    ///   <see cref="TransType.OnlyConj"/>: no transpose;<br />
    /// - <see cref="TransType.OnlyTrans"/> or 
    ///   <see cref="TransType.ConjTrans"/>: transpose.
    /// </param>
    /// <param name="alpha">Scaling factor applied to <paramref name="A"/>.</param>
    /// <param name="A">Input matrix A.</param>
    /// <param name="B">Output matrix B.</param>
    /// <remarks>
    /// If <paramref name="aTrans"/> requests a transpose,
    /// the transposed matrix <paramref name="A"/> must be conformable
    /// with <paramref name="B"/>.
    /// </remarks>
    public static void Scal2(DiagType aDiag, UpLo aUplo,
        TransType aTrans, scalar alpha, in matrix A, in matrix B)
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
                    var start = i;
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
                    var end = i + 1;
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
    /// Performs an element-wise matrix scaling operation and stores the
    /// result in another matrix.<br />
    /// <paramref name="B"/> := <paramref name="A"/> 
    /// * <paramref name="alpha"/>.
    /// </summary>
    /// <param name="alpha">Scaling factor applied to <paramref name="A"/>.</param>
    /// <param name="A">Input matrix A.</param>
    /// <param name="B">Output matrix B.</param>
    public static void Scal2(scalar alpha, in matrix A, in matrix B)
        => Scal2(DiagType.NonUnit, UpLo.Dense, TransType.NoTrans,
            alpha, A, B);

    /// <summary>
    /// Performs an element-wise matrix assignment operation.<br />
    /// Uplo?(<paramref name="A"/>) := <paramref name="alpha"/>, <br />
    /// using <paramref name="aDiag"/> and <paramref name="aUplo"/> to specify
    /// the region to be assigned.
    /// </summary>
    /// <param name="aDiag">
    /// When <paramref name="aUplo"/> is not <see cref="UpLo.Dense"/>,
    /// specifies whether the diagonal elements are assigned:<br />
    /// - <see cref="DiagType.NonUnit"/>: assign diagonal elements as usual;<br />
    /// - <see cref="DiagType.Unit"/>: skip diagonal elements when assigning.
    /// </param>
    /// <param name="aUplo">
    /// Specifies the region of the matrix to be processed:<br />
    /// - <see cref="UpLo.Dense"/>: the entire matrix;<br />
    /// - <see cref="UpLo.Upper"/>: only the upper triangular part;<br />
    /// - <see cref="UpLo.Lower"/>: only the lower triangular part.
    /// </param>
    /// <param name="alpha">The value assigned to <paramref name="A"/>.</param>
    /// <param name="A">Output matrix A.</param>
    public static void Set
        (DiagType aDiag, UpLo aUplo, scalar alpha, in matrix A)
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
                var start = Math.Max(i, 0);
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
                var end = Math.Min(n, i + 1);
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
    /// Performs an element-wise matrix assignment operation.<br />
    /// <paramref name="A"/> := <paramref name="alpha"/>.
    /// </summary>
    /// <param name="alpha">The value assigned to <paramref name="A"/>.</param>
    /// <param name="A">Output matrix A.</param>
    public static void Set(scalar alpha, in matrix A)
        => Set(DiagType.NonUnit, UpLo.Dense, alpha, A);

    /// <summary>
    /// Performs an element-wise matrix subtraction.<br />
    /// <paramref name="B"/> := <paramref name="B"/> - 
    /// Trans?(Uplo?(<paramref name="A"/>)), <br />
    /// using <paramref name="aDiag"/> and <paramref name="aUplo"/> to specify
    /// the region on which the subtraction is performed, and
    /// <paramref name="aTrans"/> to specify whether a transposed view of
    /// <paramref name="A"/> is used.
    /// </summary>
    /// <param name="aDiag">
    /// When <paramref name="aUplo"/> is not <see cref="UpLo.Dense"/>,
    /// specifies whether the diagonal elements are included in the subtraction:<br />
    /// - <see cref="DiagType.NonUnit"/>: operate on diagonal elements as usual;<br />
    /// - <see cref="DiagType.Unit"/>: skip diagonal elements in the subtraction.
    /// </param>
    /// <param name="aUplo">
    /// Specifies the region of the matrix to be processed:<br />
    /// - <see cref="UpLo.Dense"/>: the entire matrix;<br />
    /// - <see cref="UpLo.Upper"/>: only the upper triangular part;<br />
    /// - <see cref="UpLo.Lower"/>: only the lower triangular part.
    /// </param>
    /// <param name="aTrans">
    /// Specifies whether <paramref name="A"/> is transposed:<br />
    /// - <see cref="TransType.NoTrans"/> or 
    ///   <see cref="TransType.OnlyConj"/>: no transpose;<br />
    /// - <see cref="TransType.OnlyTrans"/> or 
    ///   <see cref="TransType.ConjTrans"/>: transpose.
    /// </param>
    /// <param name="A">Input matrix A.</param>
    /// <param name="B">Input/output matrix B.</param>
    /// <remarks>
    /// If <paramref name="aTrans"/> requests a transpose,
    /// the transposed matrix <paramref name="A"/> must be conformable
    /// with <paramref name="B"/>.
    /// </remarks>
    public static void Sub(DiagType aDiag, UpLo aUplo,
        TransType aTrans, in matrix A, in matrix B)
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
                    var start = i;
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
                    var end = i + 1;
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
    /// Performs an element-wise matrix subtraction.<br />
    /// <paramref name="B"/> := <paramref name="B"/> - 
    /// Trans?(Uplo?(<paramref name="A"/>)).
    /// </summary>
    /// <param name="A">Input matrix A.</param>
    /// <param name="B">Input/output matrix B.</param>
    public static void Sub(in matrix A, in matrix B)
        => Sub(DiagType.NonUnit, UpLo.Dense,
            TransType.NoTrans, A, B);
}
