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
using System.Reflection;

namespace DDLA.BLAS.Managed;

public static partial class BlasProvider
{

    /// <summary>
    /// Performs a general matrix–vector multiplication and accumulates
    /// the result into a vector.<br />
    /// <paramref name="y"/> := <paramref name="alpha"/> * 
    /// Trans?(<paramref name="A"/>) * <paramref name="x"/> 
    /// + <paramref name="beta"/> * <paramref name="y"/>.
    /// </summary>
    /// <param name="aTrans">
    /// Specifies whether to transpose <paramref name="A"/>:<br />
    /// - <see cref="TransType.NoTrans"/> or 
    ///   <see cref="TransType.OnlyConj"/>: do not transpose;<br />
    /// - <see cref="TransType.OnlyTrans"/> or 
    ///   <see cref="TransType.ConjTrans"/>: use the transposed matrix.
    /// </param>
    /// <param name="alpha">
    /// Scaling factor alpha applied to the matrix–vector product
    /// <c>Trans?(A) * x</c>.
    /// </param>
    /// <param name="A">Input matrix A.</param>
    /// <param name="x">
    /// Input vector x whose length must match the number of columns of
    /// <paramref name="A"/> after applying <paramref name="aTrans"/>.
    /// </param>
    /// <param name="beta">
    /// Scaling factor beta applied to the current contents of
    /// <paramref name="y"/>.
    /// </param>
    /// <param name="y">
    /// Input/output vector y. Its initial value is scaled by
    /// <paramref name="beta"/> and then incremented by the result of
    /// the matrix–vector multiplication. Its length must equal the
    /// length of the result vector.
    /// </param>
    /// <exception cref="ArgumentException" />
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

    /// <summary>
    /// Performs a general matrix–vector multiplication and accumulates
    /// the result into a vector (no transpose).<br />
    /// <paramref name="y"/> := <paramref name="alpha"/> * 
    /// <paramref name="A"/> * <paramref name="x"/> 
    /// + <paramref name="beta"/> * <paramref name="y"/>.
    /// </summary>
    /// <param name="alpha">
    /// Scaling factor alpha applied to the matrix–vector product
    /// <c>A * x</c>.
    /// </param>
    /// <param name="A">
    /// Input matrix A with dimensions (rows, cols).
    /// </param>
    /// <param name="x">
    /// Input vector x whose length must equal the number of columns of
    /// <paramref name="A"/>.
    /// </param>
    /// <param name="beta">
    /// Scaling factor beta applied to the current contents of
    /// <paramref name="y"/>.
    /// </param>
    /// <param name="y">
    /// Input/output vector y whose length must equal the number of rows
    /// of <paramref name="A"/>.
    /// </param>
    /// <exception cref="ArgumentException" />
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

    /// <summary>
    /// Performs a rank-1 update (GER) of a general matrix.<br />
    /// <paramref name="A"/> := <paramref name="alpha"/> * 
    /// <paramref name="x"/> * <paramref name="y"/>^T + <paramref name="A"/>.
    /// </summary>
    /// <param name="alpha">
    /// Scaling factor alpha applied to the outer product
    /// <c>x * y^T</c>.
    /// </param>
    /// <param name="x">
    /// Input vector x whose length must equal the number of rows of
    /// <paramref name="A"/>.
    /// </param>
    /// <param name="y">
    /// Input vector y whose length must equal the number of columns of
    /// <paramref name="A"/>.
    /// </param>
    /// <param name="A">
    /// Input/output matrix A to which the rank-1 update is accumulated.
    /// </param>
    /// <exception cref="ArgumentException" />
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

    /// <summary>
    /// Performs a symmetric matrix–vector multiplication and accumulates
    /// the result into a vector (symmetric MVM).<br />
    /// <paramref name="y"/> := <paramref name="alpha"/> * 
    /// <paramref name="A"/> * <paramref name="x"/> 
    /// + <paramref name="beta"/> * <paramref name="y"/>,<br />
    /// where <paramref name="A"/> is treated as a real symmetric matrix,
    /// and only the triangle specified by <paramref name="uplo"/> is used.
    /// </summary>
    /// <param name="uplo">
    /// Specifies which triangular part of <paramref name="A"/> is stored
    /// and used as the symmetric matrix:<br />
    /// - <see cref="UpLo.Upper"/>: use only the upper triangle
    ///   (including the diagonal);<br />
    /// - <see cref="UpLo.Lower"/>: use only the lower triangle
    ///   (including the diagonal).<br />
    /// The opposite triangle is not accessed.
    /// </param>
    /// <param name="alpha">
    /// Scaling factor alpha applied to the symmetric matrix–vector
    /// product <c>A * x</c>.
    /// </param>
    /// <param name="A">
    /// Input symmetric matrix A, which must be square with dimensions
    /// (n, n).
    /// </param>
    /// <param name="x">
    /// Input vector x whose length must equal the dimension n of
    /// <paramref name="A"/>.
    /// </param>
    /// <param name="beta">
    /// Scaling factor beta applied to the current contents of
    /// <paramref name="y"/>.
    /// </param>
    /// <param name="y">
    /// Input/output vector y whose length must equal the dimension n of
    /// <paramref name="A"/>.
    /// </param>
    /// <exception cref="ArgumentException" />
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

    /// <summary>
    /// Performs a symmetric rank-1 update (SYR) of a symmetric matrix.<br />
    /// <paramref name="A"/> := <paramref name="alpha"/> * 
    /// <paramref name="x"/> * <paramref name="x"/>^T + <paramref name="A"/>,<br />
    /// where <paramref name="A"/> is treated as a real symmetric matrix
    /// and only the triangle specified by <paramref name="uplo"/> is updated.
    /// </summary>
    /// <param name="uplo">
    /// Specifies which triangular part of <paramref name="A"/> is stored
    /// and updated as the symmetric matrix:<br />
    /// - <see cref="UpLo.Upper"/>: update only the upper triangle
    ///   (including the diagonal);<br />
    /// - <see cref="UpLo.Lower"/>: update only the lower triangle
    ///   (including the diagonal).
    /// </param>
    /// <param name="alpha">
    /// Scaling factor alpha applied to the outer product
    /// <c>x * x^T</c>.
    /// </param>
    /// <param name="x">
    /// Input vector x whose length must equal the dimension n of
    /// <paramref name="A"/>.
    /// </param>
    /// <param name="A">
    /// Input/output symmetric matrix A, which must be square with
    /// dimensions (n, n).
    /// </param>
    /// <exception cref="ArgumentException" />
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

    /// <summary>
    /// Performs a symmetric rank-2 update (SYR2) of a symmetric matrix.<br />
    /// UpLo?(<paramref name="A"/>) := UpLo?(<paramref name="A"/>) +
    /// <paramref name="alpha"/> * 
    /// <paramref name="x"/> * <paramref name="y"/>^T
    /// + <paramref name="alpha"/> * <paramref name="y"/> * <paramref name="x"/>^T,<br />
    /// where <paramref name="A"/> is treated as a real symmetric matrix
    /// and only the triangle specified by <paramref name="uplo"/> is updated.
    /// </summary>
    /// <param name="uplo">
    /// Specifies which triangular part of <paramref name="A"/> is stored
    /// and updated as the symmetric matrix:<br />
    /// - <see cref="UpLo.Upper"/>: update only the upper triangle
    ///   (including the diagonal);<br />
    /// - <see cref="UpLo.Lower"/>: update only the lower triangle
    ///   (including the diagonal).
    /// </param>
    /// <param name="alpha">
    /// Scaling factor alpha applied to the rank-2 update
    /// <c>x * y^T + y * x^T</c>.
    /// </param>
    /// <param name="x">
    /// Input vector x whose length must equal the dimension n of
    /// <paramref name="A"/>.
    /// </param>
    /// <param name="y">
    /// Input vector y whose length must equal the dimension n of
    /// <paramref name="A"/>.
    /// </param>
    /// <param name="A">
    /// Input/output symmetric matrix A, which must be square with
    /// dimensions (n, n).
    /// </param>
    /// <exception cref="ArgumentException" />
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

    /// <summary>
    /// Performs a BLAS-style triangular matrix–vector multiplication.<br />
    /// <paramref name="x"/> := <paramref name="alpha"/> * 
    /// Trans?(Uplo?(<paramref name="A"/>)) * <paramref name="x"/>,<br />
    /// where the structure and transpose of the triangular matrix are
    /// specified by <paramref name="aUplo"/>, <paramref name="aTrans"/>,
    /// and <paramref name="aDiag"/>.
    /// </summary>
    /// <param name="aUplo">
    /// Specifies which part of <paramref name="A"/> is interpreted as
    /// the triangular matrix:<br />
    /// - <see cref="UpLo.Upper"/>: use the upper triangle (including
    ///   the diagonal);<br />
    /// - <see cref="UpLo.Lower"/>: use the lower triangle (including
    ///   the diagonal).
    /// </param>
    /// <param name="aTrans">
    /// Specifies whether to transpose the triangular matrix:<br />
    /// - <see cref="TransType.NoTrans"/> or 
    ///   <see cref="TransType.OnlyConj"/>: do not transpose;<br />
    /// - <see cref="TransType.OnlyTrans"/> or 
    ///   <see cref="TransType.ConjTrans"/>: transpose the triangular
    ///   matrix selected by <paramref name="aUplo"/>.
    /// </param>
    /// <param name="aDiag">
    /// Specifies whether the diagonal is treated as unit:<br />
    /// - <see cref="DiagType.NonUnit"/>: use the stored diagonal
    ///   elements;<br />
    /// - <see cref="DiagType.Unit"/>: treat the diagonal as ones and
    ///   ignore the stored values.
    /// </param>
    /// <param name="alpha">
    /// Scaling factor alpha applied to the triangular matrix–vector
    /// product.
    /// </param>
    /// <param name="A">
    /// Input square matrix A (n-by-n) containing the triangular matrix.
    /// </param>
    /// <param name="x">
    /// Input/output vector x whose length must equal the dimension n of
    /// <paramref name="A"/>.
    /// </param>
    /// <exception cref="ArgumentException" />
    public static void TrMV(UpLo aUplo, TransType aTrans,
        DiagType aDiag, scalar alpha, in matrix A, in vector x)
    {
        int length = CheckSymmMatLength(A, aUplo);
        CheckLength(x, length);

        bool upper = aUplo is UpLo.Upper;
        int rs = A.RowStride;
        int cs = A.ColStride;
        if (aTrans.HasFlag(TransType.OnlyTrans))
        {
            upper = !upper;
            (rs, cs) = (cs, rs);
        }

        Details.TrMV_Impl(upper, aDiag, length, alpha, 
            ref A.GetHeadRef(), rs, cs, ref x.GetHeadRef(), x.Stride);
    }

    public static partial class Details
    {
        public static void TrMV_Impl(bool upper, DiagType aDiag, int length, scalar alpha, ref scalar aHead, int rowStride, int colStride, ref scalar yHead, int yStride)
        {
            if (length == 0)
                return;

            if (upper)
            {
                if (colStride == 1)
                    TrMV_Kernel_UpRow_Vector256(aDiag, length, alpha, ref aHead, rowStride, ref yHead, yStride);
                else if (rowStride == 1)
                    TrMV_Kernel_UpCol_Vector256(aDiag, length, alpha, ref aHead, colStride, ref yHead, yStride);
                else
                    TrMV_Kernel_Upp(aDiag, length, alpha, ref aHead, rowStride, colStride, ref yHead, yStride);
            }
            else
            {
                if (colStride == 1)
                    TrMV_Kernel_LoRow_Vector256(aDiag, length, alpha, ref aHead, rowStride, ref yHead, yStride);
                else if (rowStride == 1)
                    TrMV_Kernel_LoCol_Vector256(aDiag, length, alpha, ref aHead, colStride, ref yHead, yStride);
                else
                    TrMV_Kernel_Low(aDiag, length, alpha, ref aHead, rowStride, colStride, ref yHead, yStride);
            }
        }

        public static void TrMV_Kernel_LoRow_Vector256(DiagType aDiag, int length, scalar alpha, ref scalar aHead, int rowStride, ref scalar yHead, int yStride)
        {
            using var xBuffer = new BufferDVectorSpan(ref yHead, length, yStride, shouldCopyBack: true);
            ref var xHead = ref xBuffer.bufferHead;
            int prefer = DotxF_Kernel_RowMajor_Vector256_PerferredCount;
            ref var xRef = ref Unsafe.Add(ref xHead, length);
            var diagStride = rowStride + 1;
            ref var aRef = ref Unsafe.Add(ref aHead, 
                (length - prefer) * diagStride);
            int i = length;
            while (i > prefer)
            {
                // Y1 += alpha * A11 * X1
                // Y1 += alpha * A10 * X0
                i -= prefer;
                xRef = ref Unsafe.Subtract(ref xRef, prefer);
                TrMV_Kernel_Low_Inner(aDiag, prefer, alpha,
                    ref aRef, rowStride, 1, ref xRef);
                DotxF_Kernel_RowMajor_Vector256_Perferred_4p4(i, alpha,
                    ref Unsafe.Add(ref aHead, i * rowStride), rowStride,
                    ref xHead, 1, 1.0, ref xRef, 1);
                aRef = ref Unsafe.Subtract(ref aRef, prefer * diagStride);
            }
            if (i > 0)
            {
                TrMV_Kernel_Low_Inner(aDiag, i, alpha,
                    ref aHead, rowStride, 1, ref xHead);
            }
        }

        public static void TrMV_Kernel_LoCol_Vector256(DiagType aDiag, int length, scalar alpha, ref scalar aHead, int colStride, ref scalar yHead, int yStride)
        {
            using var xBuffer = new BufferDVectorSpan(ref yHead, length, yStride, shouldCopyBack: true);
            ref var xHead = ref xBuffer.bufferHead;
            int prefer = AxpyF_Kernel_ColMajor_Vector256_PerferredCount;
            int last = length % prefer;
            int i = length - last;
            ref var xRef = ref Unsafe.Add(ref xHead, i);
            var diagStride = colStride + 1;
            ref var aRef = ref Unsafe.Add(ref aHead, i * (colStride + 1));
            if (last > 0)
            {
                TrMV_Kernel_Low_Inner(aDiag, last, alpha, 
                    ref aRef, 1, colStride, ref xRef);
            }
            while (i > 0)
            {
                // Y2 += alpha * A21 * X1
                // Y1 += alpha * A11 * X1
                ref var xRef2 = ref xRef;
                i -= prefer;
                xRef = ref Unsafe.Subtract(ref xRef, prefer);
                aRef = ref Unsafe.Subtract(ref aRef, prefer * colStride);
                AxpyF_Kernel_ColMajor_Vector256_Perferred_8p4(last, alpha,
                    ref aRef, colStride, ref xRef, 1, ref xRef2, 1);
                aRef = ref Unsafe.Subtract(ref aRef, prefer);
                TrMV_Kernel_Low_Inner(aDiag, prefer, alpha,
                    ref aRef, 1, colStride, ref xRef);
                last += prefer;
            }
        }

        public static void TrMV_Kernel_Low(DiagType aDiag, int length, scalar alpha, ref scalar aHead, int rowStride, int colStride, ref scalar yHead, int yStride)
        {
            using var xSpan = new BufferDVectorSpan(ref yHead, length, yStride, shouldCopyBack: true);
            TrMV_Kernel_Low_Inner(aDiag, length, alpha, ref aHead, rowStride, colStride, ref xSpan.bufferHead);
        }

        public static void TrMV_Kernel_Low_Inner(DiagType aDiag, int length, scalar alpha, ref scalar aHead, int rowStride, int colStride, ref scalar yHead)
        {
            bool unit = aDiag is DiagType.Unit;
            for (int i = length - 1; i >= 0; i--)
            {
                var a = unit ? 1.0 : Unsafe.Add(ref aHead,
                    i * rowStride + i * colStride);
                var x = Unsafe.Add(ref yHead, i);
                scalar temp = a * x;
                for (int j = 0; j < i; j++)
                {
                    a = Unsafe.Add(ref aHead,
                        i * rowStride + j * colStride);
                    x = Unsafe.Add(ref yHead, j);
                    temp += a * x;
                }
                Unsafe.Add(ref yHead, i) = alpha * temp;
            }
        }

        public static void TrMV_Kernel_UpRow_Vector256(DiagType aDiag, int length, scalar alpha, ref scalar aHead, int rowStride, ref scalar yHead, int yStride)
        {
            using var xBuffer = new BufferDVectorSpan(ref yHead, length, yStride, shouldCopyBack: true);
            ref var xHead = ref xBuffer.bufferHead;
            int prefer = DotxF_Kernel_RowMajor_Vector256_PerferredCount;
            int i = 0;
            ref var xRef = ref xHead;
            ref var aRef = ref aHead;
            int last = length;
            while (i <= length - prefer)
            {
                // Y1 += alpha * A11 * X1
                // Y1 += alpha * A12 * X2
                last -= prefer;
                TrMV_Kernel_Upp_Inner(aDiag, prefer, alpha,
                    ref aRef, rowStride, 1, ref xRef);
                ref var xRef2 = ref Unsafe.Add(ref xRef, prefer);
                aRef = ref Unsafe.Add(ref aRef, prefer);
                DotxF_Kernel_RowMajor_Vector256_Perferred_4p4(last, alpha,
                    ref aRef, rowStride,
                    ref xRef2, 1, 1.0, ref xRef, 1);
                xRef = ref xRef2;
                aRef = ref Unsafe.Add(ref aRef, prefer * rowStride);
                i += prefer;
            }
            if (last > 0)
            {
                TrMV_Kernel_Upp_Inner(aDiag, last, alpha,
                    ref aRef, rowStride, 1, ref xRef);
            }
        }

        public static void TrMV_Kernel_UpCol_Vector256(DiagType aDiag, int length, scalar alpha, ref scalar aHead, int colStride, ref scalar yHead, int yStride)
        {
            using var xBuffer = new BufferDVectorSpan(ref yHead, length, yStride, shouldCopyBack: true);
            ref var xHead = ref xBuffer.bufferHead;
            int prefer = AxpyF_Kernel_ColMajor_Vector256_PerferredCount;
            int prev = length % prefer;
            int i = prev;
            ref var xRef = ref Unsafe.Add(ref xHead, i);
            ref var aRef = ref Unsafe.Add(ref aHead, i * (colStride + 1));
            if (prev > 0)
            {
                TrMV_Kernel_Upp_Inner(aDiag, prev, alpha,
                    ref aHead,
                    1, colStride, ref xHead);
            }
            while (i < length)
            {
                // Y0 += alpha * A01 * X1
                // Y1 += alpha * A11 * X1
                aRef = ref Unsafe.Add(ref aHead, i * colStride);
                AxpyF_Kernel_ColMajor_Vector256_Perferred_8p4(i, alpha,
                    ref aRef, colStride, ref xRef, 1, ref xHead, 1);
                aRef = ref Unsafe.Add(ref aRef, i);
                TrMV_Kernel_Upp_Inner(aDiag, prefer, alpha,
                    ref aRef, 1, colStride, ref xRef);
                xRef = ref Unsafe.Add(ref xRef, prefer);
                i += prefer;
            }
        }

        public static void TrMV_Kernel_Upp(DiagType aDiag, int length, scalar alpha, ref scalar aHead, int rowStride, int colStride, ref scalar yHead, int yStride)
        {
            using var xSpan = new BufferDVectorSpan(ref yHead, length, yStride, shouldCopyBack: true);
            TrMV_Kernel_Upp_Inner(aDiag, length, alpha, ref aHead, rowStride, colStride, ref xSpan.bufferHead);
        }

        public static void TrMV_Kernel_Upp_Inner(DiagType aDiag, int length, scalar alpha, ref scalar aHead, int rowStride, int colStride, ref scalar yHead)
        {
            bool unit = aDiag is DiagType.Unit;
            for (int i = 0; i < length; i++)
            {
                var a = unit ? 1.0 : Unsafe.Add(ref aHead,
                    i * (rowStride + colStride));
                var x = Unsafe.Add(ref yHead, i);
                scalar temp = a * x;
                for (int j = i + 1; j < length; j++)
                {
                    a = Unsafe.Add(ref aHead,
                        i * rowStride + j * colStride);
                    x = Unsafe.Add(ref yHead, j);
                    temp += a * x;
                }
                Unsafe.Add(ref yHead, i) = alpha * temp;
            }
        }
    }

    /// <summary>
    /// Performs a triangular matrix–vector multiplication (simple
    /// interface).<br />
    /// <paramref name="x"/> := <paramref name="alpha"/> * 
    /// T * <paramref name="x"/>, where T is the triangular part of a
    /// square matrix specified by <paramref name="uplo"/>.
    /// </summary>
    /// <param name="uplo">
    /// Specifies which part of <paramref name="A"/> is interpreted as
    /// the triangular matrix T:<br />
    /// - <see cref="UpLo.Upper"/>: use the upper triangle (including
    ///   the diagonal) and ignore the lower triangle;<br />
    /// - <see cref="UpLo.Lower"/>: use the lower triangle (including
    ///   the diagonal) and ignore the upper triangle.
    /// </param>
    /// <param name="alpha">
    /// Scaling factor alpha applied to the triangular matrix–vector
    /// product <c>T * x</c>.
    /// </param>
    /// <param name="A">
    /// Input square matrix A (n-by-n) containing the triangular matrix.
    /// </param>
    /// <param name="x">
    /// Input/output vector x whose length must equal the dimension n of
    /// <paramref name="A"/>.
    /// </param>
    /// <exception cref="ArgumentException" />
    public static void TrMV(UpLo uplo, scalar alpha,
        in matrix A, in vector x)
    {
        int length = A.Rows;
        if (length != A.Cols)
            throw new ArgumentException($"Matrix A must be square. Rows: {length}, Cols: {A.Cols}.");
        if (x.Length != length)
            throw new ArgumentException($"Length of vector must be equal to the length from one dim of matrix.");

        Details.TrMV_Impl(uplo is UpLo.Upper, DiagType.NonUnit, length, alpha, ref A.GetHeadRef(), A.RowStride, A.ColStride, ref x.GetHeadRef(), x.Stride);
    }

    /// <summary>
    /// Performs a BLAS-style triangular system solve.<br />
    /// <paramref name="y"/> := <paramref name="alpha"/> * 
    /// inv(Trans?(Uplo?(<paramref name="A"/>))) * <paramref name="y"/>,<br />
    /// where the structure and transpose of the triangular matrix are
    /// specified by <paramref name="aUplo"/>, <paramref name="aTrans"/>,
    /// and <paramref name="aDiag"/>.
    /// </summary>
    /// <param name="aUplo">
    /// Specifies which part of <paramref name="A"/> is interpreted as
    /// the triangular matrix:<br />
    /// - <see cref="UpLo.Upper"/>: use the upper triangle (including
    ///   the diagonal);<br />
    /// - <see cref="UpLo.Lower"/>: use the lower triangle (including
    ///   the diagonal).
    /// </param>
    /// <param name="aTrans">
    /// Specifies whether to transpose the triangular matrix:<br />
    /// - <see cref="TransType.NoTrans"/> or 
    ///   <see cref="TransType.OnlyConj"/>: do not transpose;<br />
    /// - <see cref="TransType.OnlyTrans"/> or 
    ///   <see cref="TransType.ConjTrans"/>: transpose the triangular
    ///   matrix selected by <paramref name="aUplo"/>.
    /// </param>
    /// <param name="aDiag">
    /// Specifies whether the diagonal is treated as unit:<br />
    /// - <see cref="DiagType.NonUnit"/>: use the stored diagonal
    ///   elements;<br />
    /// - <see cref="DiagType.Unit"/>: treat the diagonal as ones and
    ///   ignore the stored values.
    /// </param>
    /// <param name="alpha">
    /// Global scaling factor alpha applied to the solution vector.
    /// </param>
    /// <param name="A">
    /// Input square matrix A (n-by-n) containing the triangular matrix.
    /// </param>
    /// <param name="y">
    /// Input/output vector y. Initially it stores the right-hand side;
    /// after solving and scaling, it is overwritten by the solution.
    /// Its length must equal the dimension n of <paramref name="A"/>.
    /// </param>
    /// <exception cref="ArgumentException" />
    public static void TrSV(UpLo aUplo, TransType aTrans, 
        DiagType aDiag, scalar alpha, in matrix A, in vector y)
    {
        int length = CheckSymmMatLength(A, aUplo);
        CheckLength(y, length);

        bool upper = aUplo is UpLo.Upper;
        int rs = A.RowStride;
        int cs = A.ColStride;
        if (aTrans.HasFlag(TransType.OnlyTrans))
        {
            upper = !upper;
            (rs, cs) = (cs, rs);
        }

        Details.TrSV_Impl(upper, aDiag, alpha, length, 
            ref A.GetHeadRef(), rs, cs, ref y.GetHeadRef(), y.Stride);
    }

    public static partial class Details
    {
        public static void TrSV_Impl(bool upper, DiagType aDiag, scalar alpha, int length, ref scalar aHead, int rowStride, int colStride, ref scalar yHead, int yStride)
        {
            if (length == 0)
                return;

            if (upper)
            {
                if (colStride == 1)
                    TrSV_Kernel_UpRow_Vector256(aDiag, length, alpha, ref aHead, rowStride, ref yHead, yStride);
                else if (rowStride == 1)
                    TrSV_Kernel_UpCol_Vector256(aDiag, length, alpha, ref aHead, colStride, ref yHead, yStride);
                else
                    TrSV_Kernel_Upp(aDiag, length, alpha, ref aHead, rowStride, colStride, ref yHead, yStride);
            }
            else
            {
                if (colStride == 1)
                    TrSV_Kernel_LoRow_Vector256(aDiag, length, alpha, ref aHead, rowStride, ref yHead, yStride);
                else if (rowStride == 1)
                    TrSV_Kernel_LoCol_Vector256(aDiag, length, alpha, ref aHead, colStride, ref yHead, yStride);
                else
                    TrSV_Kernel_Low(aDiag, length, alpha, ref aHead, rowStride, colStride, ref yHead, yStride);
            }
        }

        public static void TrSV_Kernel_LoRow_Vector256(DiagType aDiag, int length, scalar alpha, ref scalar aHead, int rowStride, ref scalar yHead, int yStride)
        {
            using var xBuffer = new BufferDVectorSpan(ref yHead, length, yStride, alpha, shouldCopyBack: true);
            ref var xHead = ref xBuffer.bufferHead;

            int prefer = DotxF_Kernel_ColMajor_Vector256_PerferredCount;
            int prev = length % prefer;
            int i = prev;
            ref var xRef = ref Unsafe.Add(ref xHead, i);
            var diagStride = rowStride + 1;
            ref var aRef = ref Unsafe.Add(ref aHead, i * diagStride);
            if (prev > 0)
            {
                TrSV_Kernel_Low_Inner(aDiag, prev, ref aHead, 
                      rowStride, colStride: 1, ref xHead);
                //aHead = ref Unsafe.Add(ref aHead, rowStride * prev);
            }
            for (; i < length; i += prefer)
            {
                // Y1 -= A10 * X0
                // X1  = inv(A11) * Y1
                DotxF_Kernel_ColMajor_Vector256_Perferred_4p6
                    (rows: i, alpha: -1.0,
                    ref Unsafe.Add(ref aHead, i * rowStride), rowStride,
                    ref xHead, xStride: 1, beta: 1.0,
                    yHead: ref xRef, yStride: 1);
                TrSV_Kernel_Low_Inner(aDiag, prefer,
                    ref aRef, rowStride, colStride: 1, ref xRef);
                aRef = ref Unsafe.Add(ref aRef, prefer * diagStride);
                //aHead = ref Unsafe.Add(ref aHead, rowStride * prefer);
                xRef = ref Unsafe.Add(ref xRef, prefer);
            }
        }

        public static void TrSV_Kernel_LoCol_Vector256(DiagType aDiag, int length, scalar alpha, ref scalar aHead, int colStride, ref scalar yHead, int yStride)
        {
            using var xBuffer = new BufferDVectorSpan(ref yHead, length, yStride, alpha, shouldCopyBack: true);
            ref var xHead = ref xBuffer.bufferHead;
            ref var xRef = ref xBuffer.bufferHead;
            ref var yRef = ref xBuffer.bufferHead;

            int prefer = AxpyF_Kernel_ColMajor_Vector256_PerferredCount;
            int i = 0;
            int last = length;
            for (; i <= length - prefer; i += prefer)
            {
                TrSV_Kernel_Low_Inner
                      (aDiag, prefer, ref aHead, rowStride: 1, colStride, ref xRef);
                aHead = ref Unsafe.Add(ref aHead, prefer);
                yRef = ref Unsafe.Add(ref yRef, prefer);
                last -= prefer;
                AxpyF_Kernel_ColMajor_Vector256_Perferred_8p4(
                    last, alpha: -1.0,
                    ref aHead, colStride,
                    ref xRef, xStride: 1,
                    ref yRef, yStride: 1);
                aHead = ref Unsafe.Add(ref aHead, colStride * prefer);
                xRef = ref yRef;
            }
            int size = length - i;
            if (size > 0)
            {
                TrSV_Kernel_Low_Inner
                      (aDiag, last, ref aHead, rowStride: 1, colStride, ref xRef);
            }
        }

        public static void TrSV_Kernel_Low(DiagType aDiag, int length, scalar alpha,
            ref scalar aHead, int rowStride, int colStride, ref scalar yHead, int yStride)
        {
            using var xSpan = new BufferDVectorSpan(ref yHead, length, yStride, alpha, shouldCopyBack: true);
            TrSV_Kernel_Low_Inner(aDiag, length, ref aHead, rowStride, colStride, ref xSpan.bufferHead);
        }

        public static void TrSV_Kernel_Low_Inner(DiagType aDiag, int length,
            ref scalar aHead, int rowStride, int colStride, ref scalar xHead)
        {
            bool unit = aDiag is DiagType.Unit;
            double tmp = 0.0;
            ref var xRef = ref xHead;
            ref var aDiagRef = ref aHead;
            int diagStride = rowStride + colStride;
            for (int i = 0; i < length; i++)
            {
                if (unit)
                {
                    tmp = aDiagRef;
                    aDiagRef = 1.0;
                    DotV_Impl(ref aHead, ref xHead, new(i, colStride, 1), out var rho);
                    xRef -= rho;
                    aDiagRef = tmp;
                }
                else
                {
                    DotV_Impl(ref aHead, ref xHead, new(i, colStride, 1), out var rho);
                    xRef -= rho;
                    xRef /= aDiagRef;
                }
                aHead = ref Unsafe.Add(ref aHead, rowStride);
                aDiagRef = ref Unsafe.Add(ref aDiagRef, diagStride);
                xRef = ref Unsafe.Add(ref xRef, 1);
            }
        }

        public static void TrSV_Kernel_UpRow_Vector256(DiagType aDiag, int length, scalar alpha, ref scalar aHead, int rowStride, ref scalar yHead, int yStride)
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
                  (aDiag, preIter, ref aRef, rowStride, colStride: 1, ref xRef);
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
                    (aDiag, pref, ref aRef, rowStride, colStride: 1, ref xRef);
                aRef = ref Unsafe.Subtract(ref aRef, rowStride * pref);
                yRef = ref Unsafe.Subtract(ref yRef, pref);
            }
        }

        public static void TrSV_Kernel_UpCol_Vector256(DiagType aDiag, int length, scalar alpha, ref scalar aHead, int colStride, ref scalar yHead, int yStride)
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
                TrSV_Kernel_Upp_Inner(aDiag, pref, ref Unsafe.Add(ref aRef, mulLength), rowStride: 1, colStride, ref xRef);
                AxpyF_Kernel_ColMajor_Vector256_Perferred_8p4(
                    mulLength, -1.0,
                    ref aRef, colStride,
                    xHead: ref xRef, xStride: 1,
                    yHead: ref xHead, yStride: 1);
                xRef = ref Unsafe.Subtract(ref xRef, pref);
                aRef = ref Unsafe.Subtract(ref aRef, colStride * pref);
            }
            TrSV_Kernel_Upp_Inner(aDiag, mulLength, ref aHead, rowStride: 1, colStride, ref xHead);
        }

        public static void TrSV_Kernel_Upp(DiagType aDiag, int length, scalar alpha, ref scalar aHead, int rowStride, int colStride, ref scalar yHead, int yStride)
        {
            using var xSpan = new BufferDVectorSpan(ref yHead, length, yStride, alpha, shouldCopyBack: true);
            TrSV_Kernel_Upp_Inner(aDiag, length, ref aHead, rowStride, colStride, ref xSpan.bufferHead);
        }

        public static void TrSV_Kernel_Upp_Inner(DiagType aDiag, int length,
            ref scalar aHead, int rowStride, int colStride, ref scalar xHead)
        {
            bool unit = aDiag is DiagType.Unit;
            double tmp = 0.0;
            int diagStride = rowStride + colStride;
            int index = length - 1;
            ref var aDiagRef = ref Unsafe.Add(ref aHead, index * diagStride);
            ref var xRef = ref Unsafe.Add(ref xHead, index);
            ref var xOldRef = ref Unsafe.Add(ref xRef, 1);
            for (int i = 0; i < length; i++)
            {
                if (unit)
                {
                    tmp = aDiagRef;
                    aDiagRef = 1.0;
                    DotV_Impl(ref Unsafe.Add(ref aDiagRef, colStride),
                        yHead: ref xOldRef, new(i, colStride, 1),
                        out var rho);
                    xOldRef = ref xRef;
                    xRef -= rho;
                    xRef /= aDiagRef;
                    aDiagRef = tmp;
                }
                else
                {
                    DotV_Impl(ref Unsafe.Add(ref aDiagRef, colStride),
                        yHead: ref xOldRef, new(i, colStride, 1),
                        out var rho);
                    xOldRef = ref xRef;
                    xRef -= rho;
                    xRef /= aDiagRef;
                }
                aDiagRef = ref Unsafe.Subtract(ref aDiagRef, diagStride);
                xRef = ref Unsafe.Subtract(ref xRef, 1);
            }
        }
    }

    /// <summary>
    /// Solves a triangular linear system and writes the result back into
    /// the vector (simple interface).<br />
    /// Equivalent to solving the system<br />
    /// T * z = <paramref name="alpha"/> * <paramref name="y"/>,<br />
    /// where T is the triangular matrix specified by
    /// <paramref name="uplo"/>, and overwriting <paramref name="y"/>
    /// with the solution z.
    /// </summary>
    /// <param name="uplo">
    /// Specifies which part of <paramref name="A"/> is interpreted as
    /// the triangular matrix T:<br />
    /// - <see cref="UpLo.Upper"/>: use the upper triangle (including
    ///   the diagonal);<br />
    /// - <see cref="UpLo.Lower"/>: use the lower triangle (including
    ///   the diagonal).
    /// </param>
    /// <param name="alpha">
    /// Scaling factor alpha applied to the right-hand side, i.e.
    /// <paramref name="y"/> is first scaled by <paramref name="alpha"/>
    /// before solving the system.
    /// </param>
    /// <param name="A">
    /// Input square matrix A (n-by-n) containing the triangular matrix.
    /// </param>
    /// <param name="y">
    /// Input/output vector y. Initially it stores the right-hand side of
    /// the system; after scaling and solving, it is overwritten by the
    /// solution vector z. Its length must equal the dimension n of
    /// <paramref name="A"/>.
    /// </param>
    /// <exception cref="ArgumentException" />
    public static void TrSV(UpLo uplo, scalar alpha, in matrix A, in vector y)
    {
        int length = A.Rows;
        if (length != A.Cols)
            throw new ArgumentException($"Matrix A must be square. Rows: {length}, Cols: {A.Cols}.");
        if (y.Length != length)
            throw new ArgumentException($"Length of vector must be equal to the length from one dim of matrix.");

        Details.TrSV_Impl(uplo is UpLo.Upper, DiagType.NonUnit, alpha, length, ref A.GetHeadRef(), A.RowStride, A.ColStride, ref y.GetHeadRef(), y.Stride);
    }
}