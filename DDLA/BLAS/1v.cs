using scalar = double;
using vector = DDLA.Core.VectorView;
using matrix = DDLA.Core.MatrixView;
using DDLA.Misc.Flags;

namespace DDLA.BLAS;

public static partial class BlasProvider
{
    /// <summary>
    /// y += Conj?(x).
    /// </summary>
    public static void Add(in vector x, in vector y)
    {
        int length = CheckLength(x, y);
        if (length == 0) return;
        Source.Add(ConjType.NoConj, length, ref x.GetHeadRef(), x.Stride, ref y.GetHeadRef(), y.Stride);
    }

    /// <summary>
    /// index = IndexOfMax(Abs(x)).
    /// </summary>
    /// <remarks><see cref="double.NaN"/> is seen as
    /// smaller than any other value.</remarks>
    public static int AMax(in vector x)
    {
        int length = x.Length;
        if (length == 0) return -1;
        Source.Amax(length, ref x.GetHeadRef(), x.Stride, out var indexDim);
        return (int)indexDim;
    }

    /// <summary>
    /// y += alpha * Conj?(x).
    /// </summary>
    public static void Axpy(scalar alpha, in vector x, in vector y)
    {
        int length = CheckLength(x, y);
        if (length == 0) return;
        Source.Axpy(ConjType.NoConj, length, alpha, ref x.GetHeadRef(), x.Stride, ref y.GetHeadRef(), y.Stride);
    }

    public static void Axpby(scalar alpha, in vector x, scalar beta, in vector y)
    {
        int length = CheckLength(x, y);
        if (length == 0) return;
        Source.Axpby(ConjType.NoConj, length, alpha, ref x.GetHeadRef(), x.Stride, in beta, ref y.GetHeadRef(), y.Stride);
    }

    public static void Copy(in vector x, in vector y)
    {
        int length = CheckLength(x, y);
        if (length == 0) return;
        Source.Copy(ConjType.NoConj, length, ref x.GetHeadRef(), x.Stride, ref y.GetHeadRef(), y.Stride);
    }

    public static scalar Dot(in vector x, in vector y)
    {
        int length = CheckLength(x, y);
        if (length == 0) return 0.0;
        Source.Dot(ConjType.NoConj, ConjType.NoConj, 
            length, 
            ref x.GetHeadRef(), x.Stride, 
            ref y.GetHeadRef(), y.Stride, 
            out double rho);
        return rho;
    }

    public static void Dotx(scalar alpha, in vector x, in vector y, scalar beta, ref scalar rho)
    {
        int length = CheckLength(x, y);
        if (length == 0) return;
        Source.Dotx(ConjType.NoConj, ConjType.NoConj, length, alpha, ref x.GetHeadRef(), x.Stride, ref y.GetHeadRef(), y.Stride, in beta, ref rho);
    }

    public static void Invert(in vector x)
    {
        int length = x.Length;
        if (length == 0) return;
        Source.Invert(length, ref x.GetHeadRef(), x.Stride);
    }

    public static void InvScal(scalar alpha, in vector x)
    {
        ArgumentOutOfRangeException.ThrowIfEqual(alpha, 0, nameof(alpha));
        int length = x.Length;
        if (length == 0) return;
        Source.Scal(ConjType.NoConj, length, 1 / alpha, ref x.GetHeadRef(), x.Stride);
    }

    public static void Scal(scalar alpha, in vector x)
    {
        int length = x.Length;
        if (length == 0) return;
        Source.Scal(ConjType.NoConj, length, alpha, ref x.GetHeadRef(), x.Stride);
    }

    public static void Scal2(scalar alpha, in vector x, in vector y)
    {
        int length = CheckLength(x, y);
        if (length == 0) return;
        Source.Scal2(ConjType.NoConj, length, alpha, ref x.GetHeadRef(), x.Stride, ref y.GetHeadRef(), y.Stride);
    }

    public static void Set(scalar alpha, in vector x)
    {
        int length = x.Length;
        if (length == 0) return;
        Source.Set(ConjType.NoConj, length, alpha, ref x.GetHeadRef(), x.Stride);
    }

    public static void Sub(in vector x, in vector y)
    {
        int length = CheckLength(x, y);
        if (length == 0) return;
        Source.Sub(ConjType.NoConj, length, ref x.GetHeadRef(), x.Stride, ref y.GetHeadRef(), y.Stride);
    }

    public static void Swap(in vector x, in vector y)
    {
        int length = CheckLength(x, y);
        if (length == 0) return;
        Source.Swap(length, ref x.GetHeadRef(), x.Stride, ref y.GetHeadRef(), y.Stride);
    }

    public static void Xpby(in vector x, scalar beta, in vector y)
    {
        int length = CheckLength(x, y);
        if (length == 0) return;
        Source.Xpby(ConjType.NoConj, length, ref x.GetHeadRef(), x.Stride, in beta, ref y.GetHeadRef(), y.Stride);
    }
}
