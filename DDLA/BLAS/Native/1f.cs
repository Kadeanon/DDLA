using ctype = double;
using System.Runtime.InteropServices;
using DDLA.Misc.Flags;

namespace DDLA.BLAS;

public static partial class Source
{
    [DllImport(BlisDLL, EntryPoint = "bli_daxpy2v")]
    public static extern void Axpy2V(
         ConjType conjx,
         ConjType conjy,
         dim_t m,
         in ctype alphax,
         in ctype alphay,
         ref ctype x, inc_t incx,
         ref ctype y, inc_t incy,
         ref ctype z, inc_t incz
 );

    [DllImport(BlisDLL, EntryPoint = "bli_ddotaxpyv")]
    public static extern void DotAxpyV(
         ConjType conjxt,
         ConjType conjx,
         ConjType conjy,
         dim_t m,
         in ctype alpha,
         ref ctype x, inc_t incx,
         ref ctype y, inc_t incy,
         ref ctype rho,
         ref ctype z, inc_t incz
 );

    [DllImport(BlisDLL, EntryPoint = "bli_daxpyf")]
    public static extern void AxpyF(
         ConjType conja,
         ConjType conjx,
         dim_t m,
         dim_t b,
         in ctype alpha,
         ref ctype a, inc_t inca, inc_t lda,
         ref ctype x, inc_t incx,
         ref ctype y, inc_t incy
 );

    [DllImport(BlisDLL, EntryPoint = "bli_ddotxf")]
    public static extern void DotxF(
         ConjType conjat,
         ConjType conjx,
         dim_t m,
         dim_t b,
         in ctype alpha,
         ref ctype a, inc_t inca, inc_t lda,
         ref ctype x, inc_t incx,
         in ctype beta,
         ref ctype y, inc_t incy
 );

    [DllImport(BlisDLL, EntryPoint = "bli_ddotxaxpyf")]
    public static extern void DotxAxpyF(
         ConjType conjat,
         ConjType conja,
         ConjType conjw,
         ConjType conjx,
         dim_t m,
         dim_t b,
         in ctype alpha,
         ref ctype a, inc_t inca, inc_t lda,
         ref ctype w, inc_t incw,
         ref ctype x, inc_t incx,
         in ctype beta,
         ref ctype y, inc_t incy,
         ref ctype z, inc_t incz
 );
}
