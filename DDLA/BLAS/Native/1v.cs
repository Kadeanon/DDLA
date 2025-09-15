using ctype = double;
using System.Runtime.InteropServices;
using DDLA.Misc.Flags;

namespace DDLA.BLAS;

public static partial class Source
{

    /// <summary>
    /// y += Conj?(x).
    /// </summary>
    [DllImport(BlisDLL, EntryPoint = "bli_daddv")]
    public static extern void Add(
         ConjType conjx,
         dim_t n,
         ref ctype x, inc_t incx,
         ref ctype y, inc_t incy
 );

    /// <summary>
    /// index = IndexOfMax(Abs(x)).
    /// </summary>
    /// <remarks><see cref="double.NaN"/> is seen as
    /// smaller than any other value.</remarks>
    [DllImport(BlisDLL, EntryPoint = "bli_damaxv")]
    public static extern void Amax(
         dim_t n,
         ref double x, inc_t incx,
         out dim_t index
 );

    /// <summary>
    ///  y += alpha * Conj?(x).
    /// </summary>
    [DllImport(BlisDLL, EntryPoint = "bli_daxpyv")]
    public static extern void Axpy(
         ConjType conjx,
         dim_t n,
         in ctype alpha,
         ref ctype x, inc_t incx,
         ref ctype y, inc_t incy
 );

    /// <summary>
    ///  y = beta * y + alpha * Conj?(x).
    /// </summary>
    [DllImport(BlisDLL, EntryPoint = "bli_daxpbyv")]
    public static extern void Axpby(
         ConjType conjx,
         dim_t n,
         in ctype alpha,
         ref ctype x, inc_t incx,
         in ctype beta,
         ref ctype y, inc_t incy
 );

    /// <summary>
    ///  y = Conj?(x);
    /// </summary>
    [DllImport(BlisDLL, EntryPoint = "bli_dcopyv")]
    public static extern void Copy(
         ConjType conjx,
         dim_t n,
         ref ctype x, inc_t incx,
         ref ctype y, inc_t incy
 );

    /// <summary>
    /// rho = conj?(x)^T * conj?(y)
    /// </summary>
    [DllImport(BlisDLL, EntryPoint = "bli_ddotv")]
    public static extern void Dot(
         ConjType conjx,
         ConjType conjy,
         dim_t n,
         ref ctype x, inc_t incx,
         ref ctype y, inc_t incy,
         out ctype rho
 );

    /// <summary>
    ///  rho = beta * rho + alpha * conj?(x)^T * conj?(y)
    /// </summary>
    [DllImport(BlisDLL, EntryPoint = "bli_ddotxv")]
    public static extern void Dotx(
         ConjType conjx,
         ConjType conjy,
         dim_t n,
         in ctype alpha,
         ref ctype x, inc_t incx,
         ref ctype y, inc_t incy,
         in ctype beta,
         ref ctype rho
 );

    /// <summary>
    ///  x = one / x
    /// </summary>
    [DllImport(BlisDLL, EntryPoint = "bli_dinvertv")]
    public static extern void Invert(
         dim_t n,
         ref ctype x, inc_t incx
 );

    /// <summary>
    ///  x = Conj?(x) / alpha
    /// </summary>
    [DllImport(BlisDLL, EntryPoint = "bli_dinvscalv")]
    public static extern void InvScal(
         ConjType conjalpha,
         dim_t n,
         in ctype alpha,
         ref ctype x, inc_t incx
 );

    /// <summary>
    ///  x = alpha * Conj?(x).
    /// </summary>
    [DllImport(BlisDLL, EntryPoint = "bli_dscalv")]
    public static extern void Scal(
         ConjType conjalpha,
         dim_t n,
         in ctype alpha,
         ref ctype x, inc_t incx
 );

    /// <summary>
    ///  y = alpha * Conj?(x).
    /// </summary>
    [DllImport(BlisDLL, EntryPoint = "bli_dscal2v")]
    public static extern void Scal2(
         ConjType conjx,
         dim_t n,
         in ctype alpha,
         ref ctype x, inc_t incx,
         ref ctype y, inc_t incy
 );

    /// <summary>
    ///  y = alpha
    /// </summary>
    [DllImport(BlisDLL, EntryPoint = "bli_dsetv")]
    public static extern void Set(
         ConjType conjalpha,
         dim_t n,
         in ctype alpha,
         ref ctype x, inc_t incx
 );

    /// <summary>
    ///  y -= Conj?(x).
    /// </summary>
    [DllImport(BlisDLL, EntryPoint = "bli_dsubv")]
    public static extern void Sub(
         ConjType conjx,
         dim_t n,
         ref ctype x, inc_t incx,
         ref ctype y, inc_t incy
 );

    /// <summary>
    /// (x, y) = (y, x)
    /// </summary>
    [DllImport(BlisDLL, EntryPoint = "bli_dswapv")]
    public static extern void Swap(
         dim_t n,
         ref ctype x, inc_t incx,
         ref ctype y, inc_t incy
 );

    /// <summary>
    ///  y = beta * y + Conj?(x).
    /// </summary>
    [DllImport(BlisDLL, EntryPoint = "bli_dxpbyv")]
    public static extern void Xpby(
         ConjType conjx,
         dim_t n,
         ref ctype x, inc_t incx,
         in ctype beta,
         ref ctype y, inc_t incy
 );
}
