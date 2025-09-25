using DDLA.Misc.Flags;
using static DDLA.BLAS.BlasProvider;

namespace SimpleExample.LAFFExercise.QRs;

/// <summary>
/// Blocked Householder QR factorization
/// with UT representation
/// </summary>
public class HHUTQR(Matrix A) : QRBase(A)
{
    public override bool IsEconomy => false;

    public const int MaxBlockSize = 128;

    public int BlockSize { get; }
        = Math.Min(MaxBlockSize, A.Cols);

    public override void Kernel()
    {
        var T = Matrix.Create(BlockSize, A.Cols);
        QRBlock(A, T);

        Q = A.EmptyLike();
        Q.ShiftDiag(1);

        // Q = H_{0} H_{1} ... H_{k-1}
        var W = Matrix.Create(BlockSize, Q.Cols);
        ApplyQ(A, T, W, Q);

        R = new(A[..A.Cols, ..]);
        Set(DiagType.Unit, UpLo.Lower, 0.0, A);
    }

    internal static void QRUnblock(MatrixView A, MatrixView T)
    {
        for(int i = 0; i < A.MinDim; i++)
        {
            ref var tau = ref T[i, i];
            var t01 = T[..i, i];
            ref var alpha = ref A[i, i];
            var a10 = A[i, ..i];
            var a21 = A[(i + 1).., i];
            var A20 = A[(i + 1).., ..i];
            var Ab1 = A[i.., i];
            var Ab2 = A[i.., (i + 1)..];

            HHQR.BuildHH(Ab1, out tau);
            HHQR.ApplyHH(Ab1, Ab2, tau);

            a10.CopyTo(t01);
            a21.LeftMul(A20, 1.0, t01);
        }
    }

    internal static void QRBlock(MatrixView A, MatrixView T)
    {
        int iNext = 0;

        for (var i = 0; i < A.MinDim; i = iNext)
        {
            iNext = Math.Min(i + MaxBlockSize, A.MinDim);

            var T1 = T[.., i..iNext];
            var T2 = T[.., iNext..];
            var AB1 = A[i.., i..iNext];
            var AB2 = A[i.., iNext..];

            QRUnblock(AB1, T1);
            ApplyQT(AB1, T1, T2, AB2);
        }
    }

    internal static void ApplyQT(MatrixView A, MatrixView T, MatrixView W, MatrixView Q)
    {
        int block = T.Rows;
        
        for (int aLeft = 0; aLeft < A.MinDim; aLeft += block)
        {
            block = Math.Min(block, A.MinDim - aLeft);
            var aRight = aLeft + block;

            ApplyQStep(A[aLeft.., aLeft..aRight],
                T[aLeft..aRight, ..block], W[..block, ..Q.Cols], 
                Q[aLeft.., ..], trans: true);
        }
    }

    internal static void ApplyQ(MatrixView A, MatrixView T, MatrixView W, MatrixView Q)
    {
        var aLeft = A.MinDim;
        var aRight = A.MinDim;
        var qLeft = A.MinDim;
        var qRight = A.MinDim;

        bool left = A.Cols % Math.Min(T.Rows, A.Cols) > 0;
        while (aLeft > 0)
        {
            var block = Math.Min(T.Rows, aLeft);
            if (left)
            {
                block = A.Cols % block;
                left = false;
            }

            aLeft -= block;

            ApplyQStep(A[aLeft.., aLeft..aRight], 
                T[..block, aLeft..aRight], W[..block, ..], 
                Q[aLeft.., ..], false);

            aRight -= block;
        }
    }

    internal static void ApplyQStep(MatrixView A, MatrixView T, MatrixView W, MatrixView Q, bool trans)
    {
        var block = A.Cols;
        var A11 = A[..block, ..];
        var A21 = A[block.., ..];
        var B1 = Q[..block, ..];
        var B2 = Q[block.., ..];

        // B -= A * T^(-1) * A^T * B

        B1.CopyTo(W);

        // X = A^T * B
        // LEFT: B -= A * T^(-1) * X

        // X = A^T * B = A11^T * B1 + A21^T * B2
        TrMM(SideType.Left, UpLo.Lower,
            TransType.OnlyTrans, DiagType.Unit,
            1, A11, W);
        A21.T.Multify(1, B2, 1, W);

        // Y = T^(-1) * X
        // LEFT: B -= A * Y
        TrSM(SideType.Left, UpLo.Upper,
            trans ? TransType.OnlyTrans : TransType.NoTrans,
            DiagType.NonUnit,
            1, T, W);

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
