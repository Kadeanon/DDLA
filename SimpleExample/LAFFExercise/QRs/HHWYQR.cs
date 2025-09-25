using DDLA.Misc.Flags;
using static DDLA.BLAS.BlasProvider;

namespace SimpleExample.LAFFExercise.QRs;

/// <summary>
/// Blocked Householder QR factorization
/// with WY representation
/// </summary>
public class HHWYQR(Matrix A) : QRBase(A)
{
    public override bool IsEconomy => true;

    public const int MaxBlockSize = 4;

    public VectorView Taus {get; } = Vector.Create(Math.Min(A.Rows, A.Cols));

    public int BlockSize { get; }
        = Math.Min(MaxBlockSize, A.Cols);

    public override void Kernel()
    {
        var S = Matrix.Create(BlockSize, A.Cols);
        QRBlock(A, S);

        Q = A.EmptyLike();
        Q.ShiftDiag(1);

        // Q = H_{0} H_{1} ... H_{k-1}
        var W = Matrix.Create(BlockSize, Q.Cols);
        ApplyQ(A, S, W, Q);

        R = new(A[..A.Cols, ..]);
        Set(DiagType.Unit, UpLo.Lower, 0.0, A);
    }

    internal static void QRUnblock(MatrixView A, MatrixView S)
    {
        for (int i = 0; i < A.MinDim; i++)
        {
            ref var tau = ref S[i, i];
            var Ab1 = A[i.., i];
            var Ab2 = A[i.., (i + 1)..];

            var S00 = S[..i, ..i];
            var s01 = S[..i, i];
            var a10 = A[i, ..i];
            var a21 = A[(i + 1).., i];
            var A20 = A[(i + 1).., ..i];

            HHQR.BuildHH(Ab1, out tau);
            HHQR.ApplyHH(Ab1, Ab2, tau);

            tau = 1 / tau;
            a10.CopyTo(s01);
            a21.LeftMul(A20, 1.0, s01);
            TrMV(UpLo.Upper, TransType.NoTrans, DiagType.NonUnit,
                -tau, S00, s01);
        }
    }

    internal static void QRBlock(MatrixView A, MatrixView S)
    {
        int iNext = 0;

        for (var i = 0; i < A.MinDim; i = iNext)
        {
            iNext = Math.Min(i + MaxBlockSize, A.MinDim);

            var S1 = S[.., i..iNext];
            var AB1 = A[i.., i..iNext];
            var AB2 = A[i.., iNext..];

            QRUnblock(AB1, S1);
            ApplyQT(AB1, S1, S[.., iNext..], AB2);
        }
    }

    internal static void ApplyQT(MatrixView A, MatrixView S, MatrixView W, MatrixView Q)
    {
        int block = S.Rows;

        for (int aLeft = 0; aLeft < A.MinDim; aLeft += block)
        {
            block = Math.Min(block, A.MinDim - aLeft);
            var aRight = aLeft + block;

            ApplyQStep(A[aLeft.., aLeft..aRight],
                S[aLeft..aRight, ..block], W[..block, ..],
                Q[aLeft.., ..], trans: true);
        }
    }

    internal static void ApplyQ(MatrixView A, MatrixView S, MatrixView W, MatrixView Q)
    {
        var aLeft = A.MinDim;
        var aRight = A.MinDim;

        bool left = A.Cols % Math.Min(S.Rows, A.Cols) > 0;
        while (aLeft > 0)
        {
            var block = Math.Min(S.Rows, aLeft);
            if (left)
            {
                block = A.Cols % block;
                left = false;
            }

            aLeft -= block;

            ApplyQStep(A[aLeft.., aLeft..aRight],
                S[..block, aLeft..aRight], W[..block, ..], 
                Q[aLeft.., ..], trans: false);

            aRight -= block;
        }
    }

    internal static void ApplyQStep(MatrixView A, MatrixView S, MatrixView W, MatrixView Q, bool trans)
    {
        var block = A.Cols;
        var A11 = A[..block, ..];
        var A21 = A[block.., ..];
        var B1 = Q[..block, ..];
        var B2 = Q[block.., ..];

        // B -= A * S * A^T * B

        B1.CopyTo(W);

        // X = A^T * B
        // LEFT: B -= A * S * X

        // X = A^T * B = A11^T * B1 + A21^T * B2
        TrMM(SideType.Left, UpLo.Lower,
            TransType.OnlyTrans, DiagType.Unit,
            1, A11, W);
        A21.T.Multify(1, B2, 1, W);

        // Y = S * X
        // LEFT: B -= A * Y
        TrMM(SideType.Left, UpLo.Upper,
            trans ? TransType.OnlyTrans : TransType.NoTrans,
            DiagType.NonUnit,
            1, S, W);

        // B -= A * Y
        // B2 -= A21 * Y
        A21.Multify(-1, W, 1, B2);

        // B1 -= A11 * Y 
        TrMM(SideType.Left, UpLo.Lower,
            TransType.NoTrans, DiagType.Unit,
            1, A11, W);

        B1.SubtractedBy(W);
    }
}
