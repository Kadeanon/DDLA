using ctype = double;
using rtype = double;
using System.Runtime.InteropServices;
using DDLA.Misc.Flags;

namespace DDLA.BLAS;

public static partial class Source
{
    public const string DLLName =
#if !NativeBlis
        "libblis.4";
#else
        "AOCL-LibBlis-Win-MT-dll";
#endif

    public const string BlisDLL = $"Native/{DLLName}";

    [DllImport(BlisDLL, EntryPoint = "bli_dasumv")]
    public static extern void Asum(
         dim_t n,
         ref ctype x, inc_t incx,
         out rtype asum);

    [DllImport(BlisDLL, EntryPoint = "bli_dnorm1m")]
    public static extern void Nrm1(
         doff_t diagoffa,
         doff_t diaga,
         UpLo uploa,
         dim_t m,
         dim_t n,
         ref ctype a, inc_t rs_a, inc_t cs_a,
         out rtype norm);

    [DllImport(BlisDLL, EntryPoint = "bli_dnormfm")]
    public static extern void NrmF(
         doff_t diagoffa,
         doff_t diaga,
         UpLo uploa,
         dim_t m,
         dim_t n,
         ref ctype a, inc_t rs_a, inc_t cs_a,
         out rtype norm);

    [DllImport(BlisDLL, EntryPoint = "bli_dnormim")]
    public static extern void NrmInf(
         doff_t diagoffa,
         doff_t diaga,
         UpLo uploa,
         dim_t m,
         dim_t n,
         ref ctype a, inc_t rs_a, inc_t cs_a,
         out rtype norm);

    [DllImport(BlisDLL, EntryPoint = "bli_dnorm1v")]
    public static extern void Nrm1(
         dim_t n,
         ref ctype x, inc_t incx,
         out rtype norm);

    [DllImport(BlisDLL, EntryPoint = "bli_dnormfv")]
    public static extern void NrmF(
         dim_t n,
         ref ctype x, inc_t incx,
         out rtype norm);

    [DllImport(BlisDLL, EntryPoint = "bli_dnormiv")]
    public static extern void NrmInf(
         dim_t n,
         ref ctype x, inc_t incx,
         out rtype norm);

    [DllImport(BlisDLL, EntryPoint = "bli_dmkherm")]
    public static extern void MkHer(
         UpLo uploa,
         dim_t m,
         ref ctype a, inc_t rs_a, inc_t cs_a
    );

    [DllImport(BlisDLL, EntryPoint = "bli_dmksymm")]
    public static extern void MkSym(
         UpLo uploa,
         dim_t m,
         ref ctype a, inc_t rs_a, inc_t cs_a);

    [DllImport(BlisDLL, EntryPoint = "bli_dmktrim")]
    public static extern void MkTri(
         UpLo uploa,
         dim_t m,
         ref ctype a, inc_t rs_a, inc_t cs_a);

    [DllImport(BlisDLL, EntryPoint = "bli_dprintv")]
    public static extern void Print(
        [MarshalAs(UnmanagedType.LPStr)] string s1,
         dim_t m,
         ref ctype x, inc_t incx,
        [MarshalAs(UnmanagedType.LPStr)] string format,
        [MarshalAs(UnmanagedType.LPStr)] string s2);

    [DllImport(BlisDLL, EntryPoint = "bli_dprintm")]
    public static extern void Print(
        [MarshalAs(UnmanagedType.LPStr)] string s1,
         dim_t m,
         dim_t n,
         ref ctype a, inc_t rs_a, inc_t cs_a,
        [MarshalAs(UnmanagedType.LPStr)] string format,
        [MarshalAs(UnmanagedType.LPStr)] string s2);

    [DllImport(BlisDLL, EntryPoint = "bli_drandv")]
    public static extern void Rand(
         dim_t n,
         ref ctype x, inc_t incx);

    [DllImport(BlisDLL, EntryPoint = "bli_drandm")]
    public static extern void Rand(
         doff_t diagoffa,
         UpLo uploa,
         dim_t m,
         dim_t n,
         ref ctype a, inc_t rs_a, inc_t cs_a);

    [DllImport(BlisDLL, EntryPoint = "bli_dsumsqv")]
    public static extern void SumSq(
        dim_t n,
        ref ctype x, inc_t incx,
        ref rtype scale,
        ref rtype sumsq);

    [DllImport(BlisDLL, EntryPoint = "bli_deqv")]
    public static extern void Eq(
        ConjType conjx,
        dim_t n,
        ref ctype x, inc_t incx,
        ref ctype y, inc_t incy,
        ref bool is_eq);

    [DllImport(BlisDLL, EntryPoint = "bli_deqm")]
    public static extern void Eq(
        doff_t diagoffa,
          DiagType diaga,
          UpLo uploa,
          TransType transa,
          dim_t m,
          dim_t n,
          ref ctype a, inc_t rs_a, inc_t cs_a,
          ref ctype b, inc_t rs_b, inc_t cs_b,
        ref bool is_eq);
}
