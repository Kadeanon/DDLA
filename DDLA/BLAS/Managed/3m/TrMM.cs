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

    private static void TrMMInner(DiagType unit, 
        int m, int n, scalar alpha,
        UpLo aUplo, matrix A, 
        UpLo bUplo, matrix B, matrix C)
    {
        var kernel = new GEMMKernel();
        var MC = kernel.mc;
        var NC = kernel.nc;
        var KC = kernel.kc;
        var MR = kernel.mr;
        var NR = kernel.nr;

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
                TrMMRightUpper(alpha, unit, A, B);
            }
            else
            {
                TrMMRightLower(m, n, alpha, unit, A, B);
            }
        }
        else if (aUplo == UpLo.Lower)
        {
            TrMMLeftLower(m, n, alpha, unit, A, B);
        }
        else
        {
            TrMMLeftUpper(m, n, alpha, unit, A, B);
        }
    }

    #region Left Lower
    private static void TrMMLeftLower(int m, int n, scalar alpha,
        DiagType unit, matrix A, matrix B)
    {
        var kernel = new GEMMKernel();
        var block = kernel.kc;

        var partA = PartitionGrid.Create(A, m, m, Quadrant.TopLeft,
            out var A00, out var A01, out var A02,
            out var A10, out var A11, out var A12,
            out var A20, out var A21, out var A22, backward: true);
        var partB = PartitionVertical.FromBottom(B,
            out var B0,
            out var B1,
            out var B2);
        while (A00.Rows > 0)
        {
            block = Math.Min(block, A00.Rows);
            using var partAStep = partA.Step(block, block);
            using var partBStep = partB.Step(block);

            for (var j = 0; j < B.Cols; j++)
            {
                var colB = B1.GetColumn(j);
                TrMV(UpLo.Lower, TransType.NoTrans, unit,
                    alpha, A11, colB);
            }
            A10.Multify(alpha, B0, 1.0, B1);
        }
    }
    #endregion Left Lower

    #region Left Upper
    private static void TrMMLeftUpper(int m, int n, scalar alpha,
        DiagType unit, matrix A, matrix B)
    {
        var kernel = new GEMMKernel();
        var kc = kernel.kc;

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
            kc = Math.Min(kc, A22.Rows);
            using var partAStep = partA.Step(kc, kc);
            using var partBStep = partB.Step(kc);

            A01.Multify(alpha, B1, 1.0, B0);
            for (var j = 0; j < B.Cols; j++)
            {
                var colB = B1.GetColumn(j);
                TrMV(UpLo.Upper, TransType.NoTrans, unit,
                    alpha, A11, colB);
            }
        }
    }
    #endregion Left Upper

    #region Right Lower
    private static void TrMMRightLower(int m, int n, scalar alpha,
        DiagType unit, matrix A, matrix B)
    {
        var kernel = new GEMMKernel();

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

            var partB = PartitionGrid.Create(B, 0, 0, Quadrant.TopLeft,
                out var B00, out var B01, out var B02,
                out var B10, out var B11, out var B12,
                out var B20, out var B21, out var B22);
            var partA1 = PartitionHorizontal.Create(A1, 0, SideType.Left,
                out var A10, out var A11, out var A12);
            while (B22.Rows > 0)
            {
                kc = Math.Min(kc, B22.Rows);
                using var partBStep = partB.Step(kc, kc);
                using var partA1Step = partA1.Step(kc);

                A11.Multify(alpha, B10, 1.0, A10);
                for (var i = 0; i < A11.Rows; i++)
                {
                    var rowA = A11.GetRow(i);
                    TrMV(UpLo.Lower, TransType.OnlyTrans, unit,
                        alpha, B11, rowA);
                }
            }
        }
    }
    #endregion Right Lower

    #region Right Upper
    private static void TrMMRightUpper(scalar alpha,
        DiagType unit, matrix A, matrix B)
    {
        var kernel = new GEMMKernel();
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

            var partB = PartitionGrid.Create(B, B.Rows, B.Cols, Quadrant.TopLeft,
                out var B00, out var B01, out var B02,
                out var B10, out var B11, out var B12,
                out var B20, out var B21, out var B22, backward: true);
            var partA1 = PartitionHorizontal.Create(A1, A1.Cols, SideType.Left,
                out var A10, out var A11, out var A12, backward: true);
            while (B00.Rows > 0)
            {
                kc = Math.Min(kc, B00.Rows);
                using var partBStep = partB.Step(kc, kc);
                using var partA1Step = partA1.Step(kc);

                A11.Multify(alpha, B12, 1.0, A12);
                for (var i = 0; i < A11.Rows; i++)
                {
                    var rowA = A11.GetRow(i);
                    TrMV(UpLo.Upper, TransType.OnlyTrans, unit,
                        alpha, B11, rowA);
                }
            }
        }
    }
    #endregion Right Upper
}
