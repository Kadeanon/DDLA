using DDLA.BLAS;
using DDLA.Core;
using DDLA.Misc;
using DDLA.Misc.Flags;

namespace DDLA.Fatorizations;

public static class LDLT
{
    internal static int BlockSize = 8;

    internal static void LDLT_Unblock(MatrixView A)
    {
        int i = 0;
        ref var a00 = ref A[i, i];
        for (; i < A.Rows - 1; i++)
        {
            a00 = ref A.AtUncheck(i, i);
            var a10 = A.SliceColUncheck(i, i + 1);
            var a11 = A.SliceSubUncheck(i + 1, i + 1);
            BlasProvider.InvScal(a00, a10);
            BlasProvider.SyR(UpLo.Lower, -a00, a10, a11);
        }
    }

    internal static void LDLT_Block(MatrixView A)
    {
        throw new NotImplementedException("LDLT factorization for block size > 1 is not implemented yet.");
        var partA = PartitionGrid.Create
            (A, 0, 0, Quadrant.TopLeft,
            out var A00, out var a01, out var A02,
            out var a10, out var a11, out var a12,
            out var A20, out var a21, out var A22);
        int block = Math.Min(BlockSize, A22.Rows);
        while (A22.Rows > 0)
        {
            using var partAStep = partA.Step(block, block);

            LDLT_Unblock(a11);
            if (A22.Rows != 0)
            {
            }
        }
    }
}
