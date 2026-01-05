using scalar = double;
using matrix = DDLA.Core.MatrixView;
using DDLA.Misc.Flags;
using DDLA.Einsum;
using DDLA.Misc;

namespace DDLA.BLAS.Managed;

public static partial class BlasProvider
{
    /// <summary>
    /// If <paramref name="aSide"/> is <see cref="SideType.Left"/>,
    /// solve Trans(<paramref name="A"/>) * X = alpha * <paramref name="B"/>, 
    /// and overwrite <paramref name="B"/> with X.
    /// <br />
    /// Or If <paramref name="aSide"/> is <see cref="SideType.Right"/>,
    /// solve X * Trans(<paramref name="A"/>) = alpha * <paramref name="B"/>, 
    /// and overwrite <paramref name="B"/> with X, 
    /// </summary>
    /// <exception cref="ArgumentException"></exception>

    public static void TrSM
        (SideType aSide, UpLo aUplo,
        TransType aTrans, DiagType aDiag,
        in scalar alpha,
        in matrix A,
        in matrix B)
    {
        var (m, n) = GetLengths(B);
        if (m == 0 || n == 0) return;
        var aLength = CheckSymmMatLength(A, aUplo);
        var k = aSide == SideType.Left ? m : n;
        if (aLength != k)
            throw new ArgumentException("Dimensions of matrixs A must be match!");

        Scal(alpha, B);

        var AEffective = A;
        var BEffective = B;
        var outMat = B;
        if (aTrans.HasFlag(TransType.OnlyTrans))
        {
            AEffective = A.T;
            aUplo = Transpose(aUplo);
        }
        if (outMat.RowStride > outMat.ColStride)
        {
            AEffective = AEffective.T;
            BEffective = BEffective.T;
            aUplo = Transpose(aUplo);
            aSide = Transpose(aSide);
        }

        if(aSide == SideType.Left)
        {
            if (aUplo == UpLo.Upper)
            {
                TrSMLeftUpperBlock(aDiag, AEffective, BEffective);
            }
            else
            {
                TrSMLeftLower(aDiag, AEffective, BEffective);
            }
        }
        else
            if (aUplo == UpLo.Upper)
        {
            TrSMRightUpperBlock(aDiag, AEffective, BEffective);
        }
        else
        {
            TrSMRightLowerBlock(aDiag, AEffective, BEffective);
        }
    }

    public static void TrSM
        (SideType aSide, UpLo aUplo,
        in scalar alpha,
        in matrix A,
        in matrix B)
        => TrSM(aSide, aUplo,
            TransType.NoTrans, DiagType.NonUnit,
            alpha, A, B);

    #region Left Lower
    private static void TrSMLeftLower(DiagType diag, matrix A, matrix B)
    {
        var kernel = new GEMMKernel();
        var kc = kernel.kc;

        var partBPanel = PartitionHorizontal
            .FromLeft(B, out var BX0, out var BX1, out var BX2);
        while (BX2.Cols > 0)
        {
            var block = Math.Min(kc, BX2.Cols);
            using var partBPanelStep = partBPanel.Step(block);

            var partA = PartitionGrid.FromTopLeft(A,
                out var A00, out var A01, out var A02,
                out var A10, out var A11, out var A12,
                out var A20, out var A21, out var A22);
            var partBBlock = PartitionVertical.FromTop(BX1,
                out var B01,
                out var B11,
                out var B21);

            while (B21.Rows > 0)
            {
                block = Math.Min(kc, B21.Rows);
                using var partBStep = partBBlock.Step(block);
                using var partAStep = partA.Step(block, block);

                TrSMLeftLowerUnblock(diag, A11, B11);
                A21.Multify(-1.0, B11, 1.0, B21);
            }
        }
    }

    private static void TrSMLeftLowerUnblock(DiagType diag, matrix A, matrix B)
    {
        for (var j = 0; j < B.Cols; j++)
        {
            var colB = B.GetColumn(j);
            TrSV(UpLo.Lower, TransType.NoTrans, diag,
                1.0, A, colB);
        }
    }
    #endregion Left Lower

    #region Left Upper
    private static void TrSMLeftUpperBlock(DiagType diag, matrix A, matrix B)
    {
        var kernel = new GEMMKernel();
        var kc = kernel.kc;

        var partBPanel = PartitionHorizontal
            .FromLeft(B, out var BX0, out var BX1, out var BX2);
        while (BX2.Cols > 0)
        {
            var block = Math.Min(kc, BX2.Cols);
            using var partBPanelStep = partBPanel.Step(block);

            var partA = PartitionGrid.FromBottomRight(A,
                out var A00, out var A01, out var A02,
                out var A10, out var A11, out var A12,
                out var A20, out var A21, out var A22);
            var partBBlock = PartitionVertical.FromBottom(BX1,
                out var B01,
                out var B11,
                out var B21);

            while (B01.Rows > 0)
            {
                block = Math.Min(kc, B01.Rows);
                using var partBStep = partBBlock.Step(block);
                using var partAStep = partA.Step(block, block);

                TrSMLeftUpperUnblock(diag, A11, B11);
                A01.Multify(-1.0, B11, 1.0, B01);
            }
        }
    }

    private static void TrSMLeftUpperUnblock(DiagType diag, matrix A, matrix B)
    {
        for (var j = 0; j < B.Cols; j++)
        {
            var colB = B.GetColumn(j);
            TrSV(UpLo.Upper, TransType.NoTrans, diag,
                1.0, A, colB);
        }
    }
    #endregion Left Upper

    #region Right Lower
    private static void TrSMRightLowerBlock(DiagType diag, matrix A, matrix B)
    {
        var kernel = new GEMMKernel();
        var kc = kernel.kc;

        var partBPanel = PartitionVertical
            .FromTop(B, out var B0X, out var B1X, out var B2X);
        while (B2X.Rows > 0)
        { 
            var block = Math.Min(kc, B2X.Rows);
            using var partBPanelStep = partBPanel.Step(block);

            var partA = PartitionGrid.FromBottomRight(A,
                out var A00, out var A01, out var A02,
                out var A10, out var A11, out var A12,
                out var A20, out var A21, out var A22);
            var partBBlock = PartitionHorizontal.FromRight(B1X,
                out var B10,
                out var B11,
                out var B12);

            while (B10.Cols > 0)
            {
                block = Math.Min(kc, B10.Cols);
                using var partBStep = partBBlock.Step(block);
                using var partAStep = partA.Step(block, block);

                TrSMRightLowerUnblock(diag, A11, B11);
                B11.Multify(-1.0, A10, 1.0, B10);
            }
        }
    }

    private static void TrSMRightLowerUnblock(DiagType diag, matrix A, matrix B)
    {
        for (var j = 0; j < B.Rows; j++)
        {
            var rowB = B.GetRow(j);
            TrSV(UpLo.Lower, TransType.OnlyTrans, diag,
                1.0, A, rowB);
        }
    }
    #endregion Right Lower

    #region Right Upper
    private static void TrSMRightUpperBlock(DiagType diag, matrix A, matrix B)
    {
        var kernel = new GEMMKernel();
        var kc = kernel.kc;

        var partBPanel = PartitionVertical
            .FromTop(B, out var B0X, out var B1X, out var B2X);
        while(B2X.Rows > 0)
        {
            var block = Math.Min(kc, B2X.Rows);
            using var partBPanelStep = partBPanel.Step(block);

            var partA = PartitionGrid.FromTopLeft(A,
                out var A00, out var A01, out var A02,
                out var A10, out var A11, out var A12,
                out var A20, out var A21, out var A22);
            var partBBlock = PartitionHorizontal.FromLeft(B1X,
                out var B10,
                out var B11,
                out var B12);

            while(B12.Cols > 0)
            {
                block = Math.Min(kc, B12.Cols);
                using var partBStep = partBBlock.Step(block);
                using var partAStep = partA.Step(block, block);

                TrSMRightUpperUnblock(diag, A11, B11);
                B11.Multify(-1.0, A12, 1.0, B12);
            }
        }
    }

    private static void TrSMRightUpperUnblock(DiagType diag, matrix A, matrix B)
    {
        for (var j = 0; j < B.Rows; j++)
        {
            var rowB = B.GetRow(j);
            TrSV(UpLo.Upper, TransType.OnlyTrans, diag,
                1.0, A, rowB);
        }
    }
    #endregion Right Upper
}
