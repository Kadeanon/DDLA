using DDLA.UFuncs.Operators;
using DDLA.UFuncs;
using DDLA.Misc;

using scalar = double;
using vector = DDLA.Core.VectorView;
using matrix = DDLA.Core.MatrixView;
using AddOperator = DDLA.UFuncs.Operators.AddOperator<double>;
using MultiplyOperator = DDLA.UFuncs.Operators.MultiplyOperator<double>;
using MultiplyAddOperator = DDLA.UFuncs.Operators.MultiplyAddOperator<double>;
using static DDLA.UFuncs.UFunc;
using System.Runtime.CompilerServices;
using DDLA.Misc.Flags;

namespace DDLA.BLAS.Managed;

public static partial class BlasProvider
{
    public static void GeMV(TransType aTrans, scalar alpha, in matrix A, in vector x, 
        scalar beta, in vector y)
    {
        var (rows, cols) = GetLengthsAfterTrans(A, aTrans);
        CheckLength(x, cols);
        CheckLength(y, rows);
        if (rows == 0 || cols == 0)
            return;

        var (rowStride, colStride) = GetStridesAfterTrans(A, aTrans);
        Details.GeMV_Impl(rows, cols, alpha, ref A.GetHeadRef(), rowStride, colStride, ref x.GetHeadRef(), x.Stride, beta, ref y.GetHeadRef(), y.Stride);
    }

    public static void GeMV(scalar alpha, in matrix A, in vector x,
        scalar beta, in vector y)
    {
        var (rows, cols) = GetLengths(A);
        CheckLength(x, cols);
        CheckLength(y, rows);
        if (rows == 0 || cols == 0)
            return;

        Details.GeMV_Impl(rows, cols, alpha, 
            ref A.GetHeadRef(), A.RowStride, A.ColStride,
            ref x.GetHeadRef(), x.Stride, beta, ref y.GetHeadRef(), y.Stride);
    }

    public static partial class Details
    {
        public static void GeMV_Impl(int rows, int cols, scalar alpha, ref scalar aHead, int aRowStride, int aColStride, ref scalar xHead, int xStride, scalar beta, ref scalar yHead, int yStride)
        {
            if (rows == 0 || cols == 0)
                return;
            else if (alpha == 0.0)
            {
                if (beta == 0.0)
                    UFunc.Details.Apply_Impl<IdentityOperator<scalar>, scalar>
                        (ref yHead, beta, new SingleIndice(cols, yStride), new());
                else
                    UFunc.Details.Map_Impl<MultiplyOperator, scalar>
                        (ref yHead, beta, new SingleIndice(cols, yStride), new());
            }
            else
            {
                using var xBuffer = new BufferDVectorSpan(ref xHead, cols, xStride, shouldCopyBack: false);
                xStride = 1;
                xHead = ref xBuffer.bufferHead;
                using var yBuffer = new BufferDVectorSpan(ref yHead, rows, yStride, shouldCopyBack: true);
                yStride = 1;
                yHead = ref yBuffer.bufferHead;
                if (aColStride == 1)
                    GeMV_Kernel_RowMajor_Vector256(rows, cols, alpha, ref aHead, aRowStride, ref xHead, beta, ref yHead);
                else if (aRowStride == 1)
                    GeMV_Kernel_ColMajor_Vector256(rows, cols, alpha, ref aHead, aColStride, ref xHead, beta, ref yHead);
                else
                    GeMV_Kernel(rows, cols, alpha, ref aHead, aRowStride, aColStride, ref xHead, beta, ref yHead);
            }
        }

        public static void GeMV_Kernel_RowMajor_Vector256(int rows, int cols, scalar alpha, ref scalar aHead, int aRowStride, ref scalar xHead, scalar beta, ref scalar yHead)
        {
            int pref = DotxF_Kernel_ColMajor_Vector256_PerferredCount;
            int i = 0;
            for (; i <= rows - pref; i += pref)
            {
                DotxF_Kernel_ColMajor_Vector256_Perferred_4p6(cols, alpha, ref aHead, aRowStride, ref xHead, 1, beta, ref yHead, yStride: 1);
                aHead = ref Unsafe.Add(ref aHead, aRowStride * pref);
                yHead = ref Unsafe.Add(ref yHead, pref);
            }
            for (; i < rows; i++)
            {
                DotV_Kernel_Vector256(cols, ref aHead, ref xHead, out var rho);
                yHead = beta * yHead + alpha * rho;
                aHead = ref Unsafe.Add(ref aHead, aRowStride);
                yHead = ref Unsafe.Add(ref yHead, 1);
            }
        }

        public static void GeMV_Kernel_ColMajor_Vector256(int rows, int cols, scalar alpha, ref scalar aHead, int colStride, ref scalar xHead, scalar beta, ref scalar yHead)
        {
            int pref = AxpyF_Kernel_ColMajor_Vector256_PerferredCount;
            UFunc.Details.Map_Impl<MultiplyOperator, scalar>
                (ref yHead, beta, new SingleIndice(rows, 1), new());
            int i = 0;
            for (; i <= cols - pref; i += pref)
            {
                AxpyF_Kernel_ColMajor_Vector256_Perferred_8p4
                    (rows, alpha, ref aHead, colStride, ref xHead, xStride: 1,
                    ref yHead, yStride: 1);
                aHead = ref Unsafe.Add(ref aHead, colStride * pref);
                xHead = ref Unsafe.Add(ref xHead, pref);
            }
            for (; i < cols; i++)
            {
                UFunc.Details.Combine_Kernel_Vector<MultiplyAddOperator, scalar>
                    (rows, ref aHead, alpha * xHead, ref yHead, new());
                aHead = ref Unsafe.Add(ref aHead, colStride);
                xHead = ref Unsafe.Add(ref xHead, 1);
            }
        }

        public static void GeMV_Kernel(int rows, int cols, scalar alpha, ref scalar aHead, int rowStride, int colStride, ref scalar xHead, scalar beta, ref scalar yHead)
        {
            for (int i = 0; i < rows; i++)
            {
                ref var aRef = ref aHead;
                ref var xRef = ref xHead;
                var ySum = 0.0;
                for (int j = 0; j < cols; j++)
                {
                    ySum += aRef * xRef;
                    aRef = ref Unsafe.Add(ref aRef, colStride);
                    xRef = ref Unsafe.Add(ref xRef, 1);
                }
                yHead *= beta;
                yHead += alpha * ySum;
                aHead = ref Unsafe.Add(ref aHead, rowStride);
                yHead = ref Unsafe.Add(ref yHead, 1);
            }
        }
    }

    public static void GeR(scalar alpha, in vector x, in vector y, in matrix A)
    {
        int rows = A.Rows;
        int cols = A.Cols;
        if (x.Length != rows || y.Length != cols)
        {
            throw new ArgumentException($"Length of x ({x.Length}) must be equal to number of rows A ({rows}).");
        }
        Details.GeR_Impl(rows, cols, alpha, ref x.GetHeadRef(), x.Stride, ref y.GetHeadRef(), y.Stride, ref A.GetHeadRef(), A.RowStride, A.ColStride);
    }

    public static partial class Details
    {
        public static void GeR_Impl(int rows, int cols, scalar alpha, ref scalar xHead, int xStride, ref scalar yHead, int yStride, ref scalar aHead, int rowStride, int colStride)
        {
            if (rows == 0 || cols == 0 || alpha == 0.0)
                return;
            else
            {
                using var xBuffer = new BufferDVectorSpan(ref xHead, rows, xStride, shouldCopyBack: false);
                xHead = ref xBuffer.bufferHead;
                using var yBuffer = new BufferDVectorSpan(ref yHead, cols, yStride, shouldCopyBack: false);
                yHead = ref yBuffer.bufferHead;
                if (rowStride == 1)
                    GeR_Kernel_ColMajor_Vector256(rows, cols, alpha, ref xHead, ref yHead, ref aHead, colStride);
                else if (colStride == 1)
                    GeR_Kernel_RowMajor_Vector256(rows, cols, alpha, ref xHead, ref yHead, ref aHead, rowStride);
                else
                    GeR_Kernel(rows, cols, alpha, ref xHead, ref yHead, ref aHead, rowStride, colStride);
            }
        }

        private static void GeR_Kernel_ColMajor_Vector256(int rows, int cols, scalar alpha, ref scalar xHead, ref scalar yHead, ref scalar aHead, int colStride)
        {
            for (int i = 0; i < cols; i++)
            {
                UFunc.Details.Combine_Kernel_Vector<MultiplyAddOperator, scalar>
                    (rows, ref xHead, alpha * yHead, ref aHead, new());
                aHead = ref Unsafe.Add(ref aHead, colStride);
                yHead = ref Unsafe.Add(ref yHead, 1);
            }
        }

        private static void GeR_Kernel_RowMajor_Vector256(int rows, int cols, scalar alpha, ref scalar xHead, ref scalar yHead, ref scalar aHead, int rowStride)
        {
            for (int i = 0; i < rows; i++)
            {
                scalar alphaX = alpha * xHead;
                UFunc.Details.Combine_Kernel_Vector<MultiplyAddOperator, scalar>
                    (cols, ref yHead, alphaX, ref aHead, new());
                aHead = ref Unsafe.Add(ref aHead, rowStride);
                xHead = ref Unsafe.Add(ref xHead, 1);
            }
        }

        public static void GeR_Kernel(int rows, int cols, scalar alpha, ref scalar xHead, ref scalar yHead, ref scalar aHead, int rowStride, int colStride)
        {
            for (int i = 0; i < rows; i++)
            {
                UFunc.Details.Combine_Kernel<MultiplyAddOperator, scalar>(ref yHead,
                    alpha * xHead, ref aHead, new(cols, 1, colStride), new());
                aHead = ref Unsafe.Add(ref aHead, rowStride);
                xHead = ref Unsafe.Add(ref xHead, 1);
            }
        }
    }

    public static void SyMV(UpLo uplo, scalar alpha, 
        in matrix A, in vector x, scalar beta, in vector y)
    {
        int length = A.Rows;
        if (length != A.Cols)
            throw new ArgumentException($"Matrix A must be square. Rows: {length}, Cols: {A.Cols}.");
        if (x.Length != length || y.Length != length)
            throw new ArgumentException($"Length of vector must be equal to the length from one dim of matrix.");

        Details.SyMV_Impl(uplo is UpLo.Upper, length, alpha, ref A.GetHeadRef(), A.RowStride, A.ColStride, ref x.GetHeadRef(), x.Stride, beta, ref y.GetHeadRef(), y.Stride);
    }

    public static partial class Details
    {
        public static void SyMV_Impl(bool upper, int length, scalar alpha, ref scalar aHead, int rowStride, int colStride, ref scalar xHead, int xStride, scalar beta, ref scalar yHead, int yStride)
        {
            if (length == 0)
                return;

            if (upper)
            {
                (rowStride, colStride) = (colStride, rowStride);
            }
            if (colStride == 1)
                SyMV_Kernel_LoRow_Vector256(length, alpha, ref aHead, rowStride, ref xHead, xStride, beta, ref yHead, yStride);
            else if (rowStride == 1)
                SyMV_Kernel_LoCol_Vector256(length, alpha, ref aHead, colStride, ref xHead, xStride, beta, ref yHead, yStride);
            else
                SyMV_Kernel_Low(length, alpha, ref aHead, rowStride, colStride, ref xHead, xStride, beta, ref yHead, yStride);
        }

        public static void SyMV_Kernel_LoCol_Vector256(int length, scalar alpha, ref scalar aHead, int colStride, ref scalar xHead, int xStride, scalar beta, ref scalar yHead, int yStride)
        {
            using var xBuffer = new BufferDVectorSpan(ref xHead, length, xStride, shouldCopyBack: false);
            xStride = 1;
            xHead = ref xBuffer.bufferHead;
            using var yBuffer = new BufferDVectorSpan(ref yHead, length, yStride, beta, shouldCopyBack: true);
            yStride = 1;
            yHead = ref yBuffer.bufferHead;
            using var yTemp = new BufferDVectorSpan(ref yHead, length, yStride, 0.0);
            ref var yTempHeader = ref yTemp.bufferHead;

            int pref = DotxAxpyF_Kernel_ColMajor_Vector256_PerferredCount;
            int i = 0;
            if (i > pref)
            {
                for (; i <= length - pref; i += pref)
                {
                    DotxAxpyF_Kernel_ColMajor_Vector256_Perferred_4p4
                        (i, alpha,
                        ref aHead, colStride,
                        wHead: ref xHead, wStride: 1,
                        xHead: ref xHead, xStride: 1,
                        beta: 1.0,
                        yHead: ref yHead, yStride: 1,
                        zHead: ref yTempHeader, zStride: 1);
                    SyMV_Kernel_Low
                        (pref, alpha,
                        ref Unsafe.Add(ref aHead, i), rowStride: 1, colStride,
                        ref xHead, xStride,
                        beta: 1.0,
                        ref yTempHeader, yStride: 1);
                    aHead = ref Unsafe.Add(ref aHead, colStride * pref);
                }
            }
            if (i < length)
            {
                int last = length - i;
                DotxAxpyF_Kernel
                    (i, last, alpha,
                    ref aHead, aRowStride: 1, colStride,
                    wHead: ref xHead, wStride: 1,
                    xHead: ref xHead, xStride: 1,
                    1.0,
                    yHead: ref yHead, yStride: 1,
                    zHead: ref yTempHeader, zStride: 1);
                SyMV_Kernel_Low
                    (last, alpha,
                        ref Unsafe.Add(ref aHead, i), rowStride: 1, colStride,
                    ref xHead, xStride,
                    beta,
                    ref yTempHeader, yStride: 1);
                aHead = ref Unsafe.Add(ref aHead, colStride * pref);
            }
            UFunc.Details.Combine_Kernel_Vector<AddOperator>
                (length, ref yTempHeader, ref yHead, default);
        }

        public static void SyMV_Kernel_LoRow_Vector256(int length, scalar alpha, ref scalar aHead, int rowStride, ref scalar xHead, int xStride, scalar beta, ref scalar yHead, int yStride)
        {
            using var xBuffer = new BufferDVectorSpan(ref xHead, length, xStride, shouldCopyBack: false);
            xStride = 1;
            xHead = ref xBuffer.bufferHead;
            using var yBuffer = new BufferDVectorSpan(ref yHead, length, yStride, beta, shouldCopyBack: true);
            yStride = 1;
            yHead = ref yBuffer.bufferHead;
            using var yTemp = new BufferDVectorSpan(ref yHead, length, yStride, 0.0);
            ref var yTempHeader = ref yTemp.bufferHead;

            int pref = DotxAxpyF_Kernel_RowMajor_Vector256_PerferredCount;
            int i = 0;
            if (i > pref)
            {
                for (; i <= length - pref; i += pref)
                {
                    DotxAxpyF_Kernel_RowMajor_Vector256_Perferred_4p4
                        (i, alpha,
                        ref aHead, rowStride,
                        wHead: ref xHead, wStride: 1,
                        xHead: ref xHead, xStride: 1,
                        beta: 1.0,
                        yHead: ref yHead, yStride: 1,
                        zHead: ref yTempHeader, zStride: 1);
                    SyMV_Kernel_Low
                        (pref, alpha,
                        ref Unsafe.Add(ref aHead, i), rowStride, colStride: 1,
                        ref xHead, xStride,
                        beta: 1.0,
                        ref yTempHeader, yStride: 1);
                    aHead = ref Unsafe.Add(ref aHead, rowStride * pref);
                }
            }
            if (i < length)
            {
                int last = length - i;
                DotxAxpyF_Kernel
                    (i, last, alpha,
                    ref aHead, rowStride, aColStride: 1,
                    wHead: ref xHead, wStride: 1,
                    xHead: ref xHead, xStride: 1,
                    1.0,
                    yHead: ref yHead, yStride: 1,
                    zHead: ref yTempHeader, zStride: 1);
                SyMV_Kernel_Low
                    (last, alpha,
                        ref Unsafe.Add(ref aHead, i), rowStride, colStride: 1,
                    ref xHead, xStride,
                    beta,
                    ref yTempHeader, yStride: 1);
                aHead = ref Unsafe.Add(ref aHead, rowStride * pref);
            }
            UFunc.Details.Combine_Kernel_Vector<AddOperator>
            (length, ref yTempHeader, ref yHead, default);
        }

        public static void SyMV_Kernel_Low(int length, scalar alpha, ref scalar aHead, int rowStride, int colStride, ref scalar xHead, int xStride, scalar beta, ref scalar yHead, int yStride)
        {
            ref scalar xRefI = ref xHead;
            ref scalar yRefI = ref yHead;
            for (int i = 0; i < length; i++)
            {
                ref scalar xRefJ = ref xHead;
                ref scalar yRefJ = ref yHead;
                ref scalar aRef = ref aHead;

                yRefI *= beta;
                for (int j = 0; j < i; j++)
                {
                    var aVal = alpha * aRef;
                    yRefI += aVal * xRefJ;
                    yRefJ += aVal * xRefI;
                    xRefJ = ref Unsafe.Add(ref xRefJ, xStride);
                    yRefJ = ref Unsafe.Add(ref yRefJ, yStride);
                    aRef = ref Unsafe.Add(ref aRef, colStride);
                }
                yRefI += alpha * aRef * xRefI;
                aHead = ref Unsafe.Add(ref aHead, rowStride);
                xRefI = ref Unsafe.Add(ref xRefI, xStride);
                yRefI = ref Unsafe.Add(ref yRefI, yStride);
            }
        }
    }

    public static void SyR(UpLo uplo, scalar alpha, in vector x, in matrix A)
    {
        int length = A.Rows;
        if (length != A.Cols)
            throw new ArgumentException($"Matrix A must be square. Rows: {length}, Cols: {A.Cols}.");
        if (x.Length != length)
            throw new ArgumentException($"Length of vector must be equal to the length from one dim of matrix.");

        Details.SyR_Impl(uplo is UpLo.Upper, length, alpha, ref A.GetHeadRef(), A.RowStride, A.ColStride, ref x.GetHeadRef(), x.Stride);
    }

    public static partial class Details
    {
        public static void SyR_Impl(bool upper, int length, scalar alpha, ref scalar aHead, int rowStride, int colStride, ref scalar xHead, int xStride)
        {
            if (length == 0)
                return;
            if (upper)
            {
                (rowStride, colStride) = (colStride, rowStride);
            }
            if (colStride == 1)
                SyR_Kernel_LoRow_Vector256(length, alpha, ref aHead, rowStride, ref xHead, xStride);
            else if (rowStride == 1)
                SyR_Kernel_LoCol_Vector256(length, alpha, ref aHead, colStride, ref xHead, xStride);
            else
                SyR_Kernel_Low(length, alpha, ref aHead, rowStride, colStride, ref xHead, xStride);
        }

        public static void SyR_Kernel_LoRow_Vector256(int length, scalar alpha, ref scalar aHead, int rowStride, ref scalar xHead, int xStride)
        {
            using var xBuffer = new BufferDVectorSpan(ref xHead, length, xStride);
            xStride = 1;
            xHead = ref xBuffer.bufferHead;
            ref var xRefI = ref xBuffer.bufferHead;
            for (int i = 1; i <= length; i++)
            {
                UFunc.Details.Combine_Kernel_Vector<MultiplyAddOperator, scalar>
                (i, ref xHead, alpha * xRefI, ref aHead, default);
                aHead = ref Unsafe.Add(ref aHead, rowStride);
                xRefI = ref Unsafe.Add(ref xRefI, 1);
            }
        }

        public static void SyR_Kernel_LoCol_Vector256(int length, scalar alpha, ref scalar aHead, int colStride, ref scalar xHead, int xStride)
        {
            using var xBuffer = new BufferDVectorSpan(ref xHead, length, xStride);
            xStride = 1;
            xHead = ref xBuffer.bufferHead;
            ref var xRefI = ref xBuffer.bufferHead;
            int diagStride = colStride + 1;
            for (int i = length; i > 0; i--)
            {
                UFunc.Details.Combine_Kernel_Vector<MultiplyAddOperator, scalar>
                    (i, ref xRefI, alpha * xRefI, ref aHead, default);
                aHead = ref Unsafe.Add(ref aHead, diagStride);
                xRefI = ref Unsafe.Add(ref xRefI, 1);
            }
        }

        public static void SyR_Kernel_Low(int length, scalar alpha, ref scalar aHead, int rowStride, int colStride, ref scalar xHead, int xStride)
        {
            using var xBuffer = new BufferDVectorSpan(ref xHead, length, xStride);
            xStride = 1;
            xHead = ref xBuffer.bufferHead;
            ref var xRefI = ref xHead;
            for (int i = 0; i < length; i++)
            {
                ref var aRefJ = ref aHead;
                ref var xRefJ = ref xHead;
                for (int j = 0; j < i; j++)
                {
                    aRefJ += alpha * xRefI * xRefJ;
                    aRefJ = ref Unsafe.Add(ref aRefJ, colStride);
                    xRefJ = ref Unsafe.Add(ref xRefJ, xStride);
                }
                aRefJ += alpha * xRefI * xRefI;
                aHead = ref Unsafe.Add(ref aHead, rowStride);
                xRefI = ref Unsafe.Add(ref xRefI, xStride);
            }
        }
    }

    public static void SyR2(UpLo uplo, scalar alpha, 
        in vector x, in vector y, in matrix A)
    {
        int length = A.Rows;
        if (length != A.Cols)
            throw new ArgumentException($"Matrix A must be square. Rows: {length}, Cols: {A.Cols}.");
        if (x.Length != length || y.Length != length)
            throw new ArgumentException($"Length of vector must be equal to the length from one dim of matrix.");

        Details.SyR2_Impl(uplo is UpLo.Upper, length, alpha, ref A.GetHeadRef(), A.RowStride, A.ColStride, ref x.GetHeadRef(), x.Stride, ref y.GetHeadRef(), y.Stride);
    }

    public static partial class Details
    {
        public static void SyR2_Impl(bool upper, int length, scalar alpha, ref scalar aHead, int rowStride, int colStride, ref scalar xHead, int xStride, ref scalar yHead, int yStride)
        {
            if (length == 0)
                return;
            if (upper)
            {
                (rowStride, colStride) = (colStride, rowStride);
            }
            if (colStride == 1)
                SyR2_Kernel_LoRow_Vector256(length, alpha, ref aHead, rowStride, ref xHead, xStride, ref yHead, yStride);
            else if (rowStride == 1)
                SyR2_Kernel_LoCol_Vector256(length, alpha, ref aHead, colStride, ref xHead, xStride, ref yHead, yStride);
            else
                SyR2_Kernel_Low(length, alpha, ref aHead, rowStride, colStride, ref xHead, xStride, ref yHead, yStride);
        }

        public static void SyR2_Kernel_LoRow_Vector256(int length, scalar alpha, ref scalar aHead, int rowStride, ref scalar xHead, int xStride, ref scalar yHead, int yStride)
        {
            using var xBuffer = new BufferDVectorSpan(ref xHead, length, xStride);
            xStride = 1;
            xHead = ref xBuffer.bufferHead;
            using var yBuffer = new BufferDVectorSpan(ref yHead, length, yStride);
            yStride = 1;
            yHead = ref yBuffer.bufferHead;
            ref var xRefI = ref xBuffer.bufferHead;
            ref var yRefI = ref yBuffer.bufferHead;
            for (int i = 1; i <= length; i++)
            {
                Axpy2V_Kernel_Vector256(i, alpha * yRefI, alpha * xRefI, ref xHead, ref yHead, ref aHead);
                aHead = ref Unsafe.Add(ref aHead, rowStride);
                xRefI = ref Unsafe.Add(ref xRefI, 1);
                yRefI = ref Unsafe.Add(ref yRefI, 1);
            }
        }

        public static void SyR2_Kernel_LoCol_Vector256(int length, scalar alpha, ref scalar aHead, int colStride, ref scalar xHead, int xStride, ref scalar yHead, int yStride)
        {
            using var xBuffer = new BufferDVectorSpan(ref xHead, length, xStride);
            xStride = 1;
            xHead = ref xBuffer.bufferHead;
            using var yBuffer = new BufferDVectorSpan(ref yHead, length, yStride);
            yStride = 1;
            yHead = ref yBuffer.bufferHead;
            ref var xRefI = ref xBuffer.bufferHead;
            ref var yRefI = ref yBuffer.bufferHead;
            int diagStride = colStride + 1;
            for (int i = length; i > 0; i--)
            {
                Axpy2V_Kernel_Vector256(i, alpha * yRefI, alpha * xRefI, ref xRefI, ref yRefI, ref aHead);
                aHead = ref Unsafe.Add(ref aHead, diagStride);
                xRefI = ref Unsafe.Add(ref xRefI, 1);
                yRefI = ref Unsafe.Add(ref yRefI, 1);
            }
        }

        public static void SyR2_Kernel_Low(int length, scalar alpha, ref scalar aHead, int rowStride, int colStride, ref scalar xHead, int xStride, ref scalar yHead, int yStride)
        {
            using var xBuffer = new BufferDVectorSpan(ref xHead, length, xStride);
            xStride = 1;
            xHead = ref xBuffer.bufferHead;
            using var yBuffer = new BufferDVectorSpan(ref yHead, length, yStride);
            yStride = 1;
            yHead = ref yBuffer.bufferHead;
            ref var xRefI = ref xHead;
            ref var yRefI = ref yHead;
            for (int i = 0; i < length; i++)
            {
                ref var aRef = ref aHead;
                ref var xRefJ = ref xHead;
                ref var yRefJ = ref yHead;
                scalar xValI = xRefI * alpha;
                scalar yValI = yRefI * alpha;
                for (int j = 0; j <= i; j++)
                {
                    scalar del = alpha * xRefI * yRefJ;
                    del += alpha * xRefJ * yRefI;
                    aRef += del;
                    aRef = ref Unsafe.Add(ref aRef, colStride);
                    xRefJ = ref Unsafe.Add(ref xRefJ, xStride);
                    yRefJ = ref Unsafe.Add(ref yRefJ, xStride);
                }
                aHead = ref Unsafe.Add(ref aHead, rowStride);
                xRefI = ref Unsafe.Add(ref xRefI, xStride);
                yRefI = ref Unsafe.Add(ref yRefI, xStride);
            }
        }
    }

    public static void TrMV(UpLo uplo, scalar alpha, in matrix A, in vector x)
    {
        int length = A.Rows;
        if (length != A.Cols)
            throw new ArgumentException($"Matrix A must be square. Rows: {length}, Cols: {A.Cols}.");
        if (x.Length != length)
            throw new ArgumentException($"Length of vector must be equal to the length from one dim of matrix.");

        Details.TrMV_Impl(uplo is UpLo.Upper, length, alpha, ref A.GetHeadRef(), A.RowStride, A.ColStride, ref x.GetHeadRef(), x.Stride);
    }

    public static partial class Details
    {
        public static void TrMV_Impl(bool upper, int length, scalar alpha, ref scalar aHead, int rowStride, int colStride, ref scalar xHead, int xStride)
        {
            if (length == 0)
                return;

            if (upper)
            {
                TrMV_Kernel_Upp(length, alpha, ref aHead, rowStride, colStride, ref xHead, xStride);
            }
            else
            {
                TrMV_Kernel_Low(length, alpha, ref aHead, rowStride, colStride, ref xHead, xStride);
            }
        }

        public static void TrMV_Kernel_Low(int length, scalar alpha, ref scalar aHead, int rowStride, int colStride, ref scalar xHead, int xStride)
        {
            for (int i = length - 1; i >= 0; i--)
            {
                scalar temp = 0.0;
                for (int j = 0; j <= i; j++)
                {
                    var A = aHead.At(i, j, rowStride, colStride);
                    var x = xHead.At(j, xStride);
                    temp += A * x;
                }
                xHead.At(i, xStride) = alpha * temp;
            }
        }

        public static void TrMV_Kernel_Upp(int length, scalar alpha, ref scalar aHead, int rowStride, int colStride, ref scalar xHead, int xStride)
        {
            for (int i = 0; i < length; i++)
            {
                scalar temp = 0.0;
                for (int j = i; j < length; j++)
                {
                    var A = aHead.At(i, j, rowStride, colStride);
                    var x = xHead.At(j, xStride);
                    temp += A * x;
                }
                xHead.At(i, xStride) = alpha * temp;
            }
        }
    }

    public static void TrSV(UpLo uplo, scalar alpha, in matrix A, in vector y)
    {
        int length = A.Rows;
        if (length != A.Cols)
            throw new ArgumentException($"Matrix A must be square. Rows: {length}, Cols: {A.Cols}.");
        if (y.Length != length)
            throw new ArgumentException($"Length of vector must be equal to the length from one dim of matrix.");

        Details.TrSV_Impl(uplo is UpLo.Upper, alpha, length, ref A.GetHeadRef(), A.RowStride, A.ColStride, ref y.GetHeadRef(), y.Stride);
    }

    public static void TrMV
        (UpLo aUplo, TransType aTrans, DiagType aDiag, scalar alpha, in matrix A, in vector x)
    {
        int length = CheckSymmMatLength(A, aUplo);
        CheckLength(x, length);

        Source.TrMV(aUplo,
            aTrans,
            aDiag,
            length,
            in alpha,
            ref A.GetHeadRef(), A.RowStride, A.ColStride,
            ref x.GetHeadRef(), x.Stride);
    }

    public static void TrSV
        (UpLo aUplo, TransType aTrans, DiagType aDiag, scalar alpha, in matrix A, in vector y)
    {
        int length = CheckSymmMatLength(A, aUplo);
        CheckLength(y, length);

        Source.TrSV(aUplo,
            aTrans,
            aDiag,
            length,
            in alpha,
            ref A.GetHeadRef(), A.RowStride, A.ColStride,
            ref y.GetHeadRef(), y.Stride);
    }
    
    public static partial class Details
    {
        public static void TrSV_Impl(bool upper, scalar alpha, int length, ref scalar aHead, int rowStride, int colStride, ref scalar yHead, int yStride)
        {
            if (length == 0)
                return;

            if (upper)
            {
                if (colStride == 1)
                    TrSV_Kernel_UpRow_Vector256(length, alpha, ref aHead, rowStride, ref yHead, yStride);
                else if (rowStride == 1)
                    TrSV_Kernel_UpCol_Vector256(length, alpha, ref aHead, colStride, ref yHead, yStride);
                else
                    TrSV_Kernel_Upp(length, alpha, ref aHead, rowStride, colStride, ref yHead, yStride);
            }
            else
            {
                if (colStride == 1)
                    TrSV_Kernel_LoRow_Vector256(length, alpha, ref aHead, rowStride, ref yHead, yStride);
                else if (rowStride == 1)
                    TrSV_Kernel_LoCol_Vector256(length, alpha, ref aHead, colStride, ref yHead, yStride);
                else
                    TrSV_Kernel_Low(length, alpha, ref aHead, rowStride, colStride, ref yHead, yStride);
            }
        }

        public static void TrSV_Kernel_LoRow_Vector256(int length, scalar alpha, ref scalar aHead, int rowStride, ref scalar yHead, int yStride)
        {
            using var xBuffer = new BufferDVectorSpan(ref yHead, length, yStride, shouldCopyBack: false);
            ref var xHead = ref xBuffer.bufferHead;

            int pref = DotxF_Kernel_ColMajor_Vector256_PerferredCount;
            int i = 0;
            int iterNums = length / pref;
            int preIter = length - pref * iterNums;
            ref var xRef = ref xHead;
            TrSV_Kernel_Low_Inner
                  (preIter, ref aHead, rowStride, colStride: 1, ref xRef);
            aHead = ref Unsafe.Add(ref aHead, rowStride * preIter);
            xRef = ref Unsafe.Add(ref xRef, preIter);
            i += preIter;
            for (; i < length; i += pref)
            {
                DotxF_Kernel_ColMajor_Vector256_Perferred_4p6
                    (rows: i, alpha: -1.0,
                    ref aHead, rowStride,
                    ref xHead, xStride: 1,
                    beta: 1.0,
                    yHead: ref xRef, yStride: 1);
                TrSV_Kernel_Low_Inner
                  (pref, ref Unsafe.Add(ref aHead, i), rowStride, colStride: 1, ref xRef);
                aHead = ref Unsafe.Add(ref aHead, rowStride * pref);
                xRef = ref Unsafe.Add(ref xRef, pref);
            }
        }

        public static void TrSV_Kernel_LoCol_Vector256(int length, scalar alpha, ref scalar aHead, int colStride, ref scalar yHead, int yStride)
        {
            using var xBuffer = new BufferDVectorSpan(ref yHead, length, yStride, alpha, shouldCopyBack: true);
            ref var xHead = ref xBuffer.bufferHead;
            ref var xRef = ref xBuffer.bufferHead;
            ref var yRef = ref xBuffer.bufferHead;

            int pref = AxpyF_Kernel_ColMajor_Vector256_PerferredCount;
            int i = 0;
            for (; i <= length - pref; i += pref)
            {
                TrSV_Kernel_Low_Inner
                      (pref, ref aHead, rowStride: 1, colStride, ref xRef);
                aHead = ref Unsafe.Add(ref aHead, pref);
                yRef = ref Unsafe.Add(ref yRef, pref);
                AxpyF_Kernel_ColMajor_Vector256_Perferred_8p4(
                    length - pref - i, alpha: -1.0,
                    ref aHead, colStride,
                    ref xRef, xStride: 1,
                    ref yRef, yStride: 1);
                aHead = ref Unsafe.Add(ref aHead, colStride * pref);
                xRef = ref yRef;
            }
            int size = length - i;
            if (size > 0)
            {
                TrSV_Kernel_Low_Inner
                      (pref, ref aHead, rowStride: 1, colStride, ref xRef);
            }
        }

        public static void TrSV_Kernel_Low(int length, scalar alpha,
            ref scalar aHead, int rowStride, int colStride, ref scalar yHead, int yStride)
        {
            using var xSpan = new BufferDVectorSpan(ref yHead, length, yStride, alpha, shouldCopyBack: true);
            TrSV_Kernel_Low_Inner(length, ref aHead, rowStride, colStride, ref xSpan.bufferHead);
        }

        public static void TrSV_Kernel_Low_Inner(int length,
            ref scalar aHead, int rowStride, int colStride, ref scalar xHead)
        {
            ref var xRef = ref xHead;
            ref var aDiagRef = ref aHead;
            int diagStride = rowStride + colStride;
            for (int i = 0; i < length; i++)
            {
                DotV_Impl(ref aHead, ref xHead, new(i, colStride, 1), out var rho);
                xRef -= rho;
                xRef /= aDiagRef;
                aHead = ref Unsafe.Add(ref aHead, rowStride);
                aDiagRef = ref Unsafe.Add(ref aDiagRef, diagStride);
                xRef = ref Unsafe.Add(ref xRef, 1);
            }
        }

        public static void TrSV_Kernel_UpRow_Vector256(int length, scalar alpha, ref scalar aHead, int rowStride, ref scalar yHead, int yStride)
        {
            using var xBuffer = new BufferDVectorSpan(ref yHead, length, yStride, alpha, shouldCopyBack: true);
            ref var xHead = ref xBuffer.bufferHead;

            int pref = DotxF_Kernel_ColMajor_Vector256_PerferredCount;
            int iterNums = length / pref;
            int preIter = length - pref * iterNums;
            ref var xRef = ref Unsafe.Add(ref xHead, pref * iterNums);
            ref var aRef = ref Unsafe.Add(ref aHead, pref * iterNums * (rowStride + 1));
            int i = 0;
            TrSV_Kernel_Upp_Inner
                  (preIter, ref aRef, rowStride, colStride: 1, ref xRef);
            i += preIter;
            aRef = ref Unsafe.Subtract(ref aRef, rowStride * pref);
            //xRef = ref Unsafe.Subtract(ref xRef, pref);
            ref var yRef = ref Unsafe.Subtract(ref xRef, pref);
            for (; i < length; i += pref)
            {
                DotxF_Kernel_ColMajor_Vector256_Perferred_4p6
                    (i, alpha: -1.0, ref aRef, rowStride,
                    ref xRef, xStride: 1,
                    beta: 1.0,
                    ref yRef, yStride: 1);
                aRef = ref Unsafe.Subtract(ref aRef, pref);
                xRef = ref yRef;
                TrSV_Kernel_Upp_Inner
                    (pref, ref aRef, rowStride, colStride: 1, ref xRef);
                aRef = ref Unsafe.Subtract(ref aRef, rowStride * pref);
                yRef = ref Unsafe.Subtract(ref yRef, pref);
            }
        }

        public static void TrSV_Kernel_UpCol_Vector256(int length, scalar alpha, ref scalar aHead, int colStride, ref scalar yHead, int yStride)
        {
            using var xBuffer = new BufferDVectorSpan(ref yHead, length, yStride, alpha, shouldCopyBack: true);
            ref var xHead = ref xBuffer.bufferHead;

            int pref = AxpyF_Kernel_ColMajor_Vector256_PerferredCount;
            int iterNums = length / pref;
            int preIter = length - pref * iterNums;
            ref var xRef = ref Unsafe.Add(ref xHead, length - pref);
            ref var aRef = ref Unsafe.Add(ref aHead, (length - pref) * colStride);
            int i = 0;
            int mulLength = length;
            for (; i <= length - pref; i += pref)
            {
                mulLength -= pref;
                TrSV_Kernel_Upp_Inner(pref, ref Unsafe.Add(ref aRef, mulLength), rowStride: 1, colStride, ref xRef);
                AxpyF_Kernel_ColMajor_Vector256_Perferred_8p4(
                    mulLength, -1.0,
                    ref aRef, colStride,
                    xHead: ref xRef, xStride: 1,
                    yHead: ref xHead, yStride: 1);
                xRef = ref Unsafe.Subtract(ref xRef, pref);
                aRef = ref Unsafe.Subtract(ref aRef, colStride * pref);
            }
            TrSV_Kernel_Upp_Inner(mulLength, ref aHead, rowStride: 1, colStride, ref xHead);
        }

        public static void TrSV_Kernel_Upp(int length, scalar alpha, ref scalar aHead, int rowStride, int colStride, ref scalar yHead, int yStride)
        {
            using var xSpan = new BufferDVectorSpan(ref yHead, length, yStride, alpha, shouldCopyBack: true);
            TrSV_Kernel_Upp_Inner(length, ref aHead, rowStride, colStride, ref xSpan.bufferHead);
        }

        public static void TrSV_Kernel_Upp_Inner(int length,
            ref scalar aHead, int rowStride, int colStride, ref scalar xHead)
        {
            int diagStride = rowStride + colStride;
            int index = length - 1;
            ref var aDiagRef = ref Unsafe.Add(ref aHead, index * diagStride);
            ref var xRef = ref Unsafe.Add(ref xHead, index);
            ref var xOldRef = ref Unsafe.Add(ref xRef, 1);
            for (int i = 0; i < length; i++)
            {
                DotV_Impl(ref Unsafe.Add(ref aDiagRef, colStride),
                    yHead: ref xOldRef, new(i, colStride, 1),
                    out var rho);
                var xRef0 = xRef;
                xOldRef = ref xRef;
                xRef -= rho;
                xRef /= aDiagRef;
                aDiagRef = ref Unsafe.Subtract(ref aDiagRef, diagStride);
                xRef = ref Unsafe.Subtract(ref xRef, 1);
            }
        }
    }

    public static ref scalar At(this ref scalar head,
        int i, int j, int rowStride, int colStride)
    {
        return ref Unsafe.Add(ref head,
            i * rowStride + j * colStride);
    }

    public static ref scalar At(this ref scalar offset,
        int x, int stride)
    {
        return ref Unsafe.Add(
            ref offset, x * stride);
    }

}
