using ctype = double;
using rtype = double;
using System.Runtime.InteropServices;
using DDLA.Misc.Flags;

namespace DDLA.BLAS;

public static partial class Source
{
    [DllImport(BlisDLL, EntryPoint = "bli_dgemv")]
    public static extern void GeMV(
         TransType transa,
         ConjType conjx,
         dim_t m,
         dim_t n,
         in ctype alpha,
         ref ctype a, inc_t rsa, inc_t csa,
         ref ctype x, inc_t incx,
         in ctype beta,
         ref ctype y, inc_t incy
 );

    [DllImport(BlisDLL, EntryPoint = "bli_dger")]
    public static extern void GeR(
         ConjType conjx,
         ConjType conjy,
         dim_t m,
         dim_t n,
         in ctype alpha,
         ref ctype x, inc_t incx,
         ref ctype y, inc_t incy,
         ref ctype a, inc_t rsa, inc_t csa
 );

    [DllImport(BlisDLL, EntryPoint = "bli_dhemv")]
    public static extern void HeMV(
         UpLo uploa,
         ConjType conja,
         ConjType conjx,
         dim_t m,
         in ctype alpha,
         ref ctype a, inc_t rsa, inc_t csa,
         ref ctype x, inc_t incx,
         in ctype beta,
         ref ctype y, inc_t incy
 );

    [DllImport(BlisDLL, EntryPoint = "bli_dher")]
    public static extern void HeR(
         UpLo uploa,
         ConjType conjx,
         dim_t m,
         ref rtype alpha,
         ref ctype x, inc_t incx,
         ref ctype a, inc_t rsa, inc_t csa
 );

    [DllImport(BlisDLL, EntryPoint = "bli_dher2")]
    public static extern void HeR2(
         UpLo uploa,
         ConjType conjx,
         ConjType conjy,
         dim_t m,
         in ctype alpha,
         ref ctype x, inc_t incx,
         ref ctype y, inc_t incy,
         ref ctype a, inc_t rsa, inc_t csa
 );

    [DllImport(BlisDLL, EntryPoint = "bli_dsymv")]
    public static extern void SyMV(
         UpLo uploa,
         ConjType conja,
         ConjType conjx,
         dim_t m,
         in ctype alpha,
         ref ctype a, inc_t rsa, inc_t csa,
         ref ctype x, inc_t incx,
         in ctype beta,
         ref ctype y, inc_t incy
 );

    [DllImport(BlisDLL, EntryPoint = "bli_dsyr")]
    public static extern void SyR(
         UpLo uploa,
         ConjType conjx,
         dim_t m,
         in ctype alpha,
         ref ctype x, inc_t incx,
         ref ctype a, inc_t rsa, inc_t csa
 );

    [DllImport(BlisDLL, EntryPoint = "bli_dsyr2")]
    public static extern void SyR2(
         UpLo uploa,
         ConjType conjx,
         ConjType conjy,
         dim_t m,
         in ctype alpha,
         ref ctype x, inc_t incx,
         ref ctype y, inc_t incy,
         ref ctype a, inc_t rsa, inc_t csa
 );

    [DllImport(BlisDLL, EntryPoint = "bli_dtrmv")]
    public static extern void TrMV(
         UpLo uploa,
         TransType transa,
         DiagType diaga,
         dim_t m,
         in ctype alpha,
         ref ctype a, inc_t rsa, inc_t csa,
         ref ctype x, inc_t incx
 );

    [DllImport(BlisDLL, EntryPoint = "bli_dtrsv")]
    public static extern void TrSV(
         UpLo uploa,
         TransType transa,
         DiagType diaga,
         dim_t m,
         in ctype alpha,
         ref ctype a, inc_t rsa, inc_t csa,
         ref ctype y, inc_t incy
 );
}
