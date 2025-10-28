using scalar = double;
using vector = DDLA.Core.VectorView;
using matrix = DDLA.Core.MatrixView;
using DDLA.Misc.Flags;
using DDLA.Einsum;
using DDLA.Utilities;
using DDLA.Misc.Pools;
using DDLA.Misc;

namespace DDLA.BLAS.Managed;

public static partial class BlasProvider 
{

    private static void TrSMInner(DiagType unit, 
        int m, int n, 
        UpLo aUplo, matrix A, 
        UpLo bUplo, matrix B, matrix C)
    {
        var kernel = new GEMMKernel();
        var MC = kernel.mc;
        var NC = kernel.nc;
        var KC = kernel.kc;
        var MR = kernel.mr;
        var NR = kernel.nr;

        var alpha = 1.0;

        var shouldTrans =
            (kernel.preferCol && C.RowStride < C.ColStride) ||
            (!kernel.preferCol && C.ColStride < C.RowStride);
        if (shouldTrans)
        {
            C = C.T;
            (m, n) = (n, m);
            (A, B) = (B.T, A.T);
            (aUplo, bUplo) = (Transpose(bUplo), Transpose(aUplo));
        }

        if (aUplo is UpLo.Dense)
        {
            if (bUplo == UpLo.Upper)
            {
                TrSMRightUpper(alpha, unit, A, B);
            }
            else
            {
                TrSMRightLower(alpha, unit, A, B);
            }
        }
        else if (aUplo == UpLo.Lower)
        {
            TrSMLeftLower(alpha, unit, A, B);
        }
        else
        {
            TrSMLeftUpper(alpha, unit, A, B);
        }
    }

    #region Left Lower
    private static void TrSMLeftLower(scalar alpha,
        DiagType unit, matrix A, matrix B)
    {
        var kernel = new GEMMKernel();
        var block = kernel.kc;

        var partA = PartitionGrid.Create(A, 0, 0, Quadrant.TopLeft,
            out var A00, out var A01, out var A02,
            out var A10, out var A11, out var A12,
            out var A20, out var A21, out var A22);
        var partB = PartitionVertical.FromTop(B,
            out var B0,
            out var B1,
            out var B2);
        while (A22.Rows > 0)
        {
            block = Math.Min(block, A22.Rows);
            using var partAStep = partA.Step(block, block);
            using var partBStep = partB.Step(block);

            A10.Multify(-1, B0, alpha, B1);
            for (var j = 0; j < B1.Cols; j++)
            {
                var colB = B1.GetColumn(j);
                TrSV(UpLo.Lower, TransType.NoTrans, unit,
                    1.0, A11, colB);
            }
        }
    }
    #endregion Left Lower

    #region Left Upper
    private static void TrSMLeftUpper(scalar alpha,
        DiagType unit, matrix A, matrix B)
    {
        var kernel = new GEMMKernel();
        var block = kernel.kc;

        var partA = PartitionGrid.FromBottomRight(A,
            out var A00, out var A01, out var A02,
            out var A10, out var A11, out var A12,
            out var A20, out var A21, out var A22);
        var partB = PartitionVertical.FromBottom(B,
            out var B0,
            out var B1,
            out var B2);
        while (A00.Rows > 0)
        {
            block = Math.Min(block, A00.Rows);
            using var partAStep = partA.Step(block, block);
            using var partBStep = partB.Step(block);

            A12.Multify(-1, B2, alpha, B1);
            for (var j = 0; j < B1.Cols; j++)
            {
                var colB = B1.GetColumn(j);
                TrSV(UpLo.Upper, TransType.NoTrans, unit,
                    1.0, A11, colB);
            }
        }
    }
    #endregion Left Upper

    #region Right Lower
    private static void TrSMRightLower(scalar alpha,
        DiagType unit, matrix A, matrix B)
    {
        var kernel = new GEMMKernel();
        var kc = kernel.kc;

        var partB = PartitionGrid.FromBottomRight(B,
            out var B00, out var B01, out var B02,
            out var B10, out var B11, out var B12,
            out var B20, out var B21, out var B22);
        var partA = PartitionHorizontal.FromRight(A,
            out var A0, out var A1, out var A2);
        while (B00.Rows > 0)
        {
            kc = Math.Min(kc, B00.Rows);
            using var partBStep = partB.Step(kc, kc);
            using var partAStep = partA.Step(kc);

            A2.Multify(-1.0, B21, alpha, A1);
            for (var i = 0; i < A1.Rows; i++)
            {
                var rowA = A1.GetRow(i);
                TrSV(UpLo.Lower, TransType.OnlyTrans, unit,
                    1.0, B11, rowA);
            }
        }
    }
    #endregion Right Lower

    #region Right Upper
    private static void TrSMRightUpper(scalar alpha,
        DiagType unit, matrix A, matrix B)
    {
        var kernel = new GEMMKernel();
        var n = A.Cols;
        var mc = kernel.mc;

        var partA = PartitionVertical.FromTop(A,
            out var A0,
            out var A1,
            out var A2);
        while (A2.Rows > 0)
        {
            mc = Math.Min(mc, A2.Rows);
            using var partAStep = partA.Step(mc);

            var kc = kernel.kc;

            var partB = PartitionGrid.FromTopLeft(B,
                out var B00, out var B01, out var B02,
                out var B10, out var B11, out var B12,
                out var B20, out var B21, out var B22);
            var partA1 = PartitionHorizontal.FromLeft(A1,
                out var A10, out var A11, out var A12);
            while (B22.Rows > 0)
            {
                kc = Math.Min(kc, B22.Rows);
                using var partBStep = partB.Step(kc, kc);
                using var partA1Step = partA1.Step(kc);

                A10.Multify(-1.0, B01, alpha, A11);
                for (var i = 0; i < A11.Rows; i++)
                {
                    var rowA = A11.GetRow(i);
                    TrSV(UpLo.Upper, TransType.OnlyTrans, unit,
                        1.0, B11, rowA);
                }
            }
        }
    }
    #endregion Right Upper
}
