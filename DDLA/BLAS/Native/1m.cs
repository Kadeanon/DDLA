using ctype = double;
using System.Runtime.InteropServices;
using DDLA.Misc.Flags;

namespace DDLA.BLAS;

public static partial class Source
{

    [DllImport(BlisDLL, EntryPoint = "bli_daddm")]
    public static extern void Add(
         doff_t diagoffa,
         DiagType diaga,
         UpLo uploa,
         TransType transa,
         dim_t m,
         dim_t n,
         ref ctype a, inc_t rsa, inc_t csa,
         ref ctype b, inc_t rsb, inc_t csb
 );

    [DllImport(BlisDLL, EntryPoint = "bli_daxpym")]
    public static extern void Axpy(
         doff_t diagoffa,
         DiagType diaga,
         UpLo uploa,
         TransType transa,
         dim_t m,
         dim_t n,
         in ctype alpha,
         ref ctype a, inc_t rsa, inc_t csa,
         ref ctype b, inc_t rsb, inc_t csb
 );

    [DllImport(BlisDLL, EntryPoint = "bli_dcopym")]
    public static extern void Copy(
         doff_t diagoffa,
         DiagType diaga,
         UpLo uploa,
         TransType transa,
         dim_t m,
         dim_t n,
         ref ctype a, inc_t rsa, inc_t csa,
         ref ctype b, inc_t rsb, inc_t csb
 );

    [DllImport(BlisDLL, EntryPoint = "bli_dinvscalm")]
    public static extern void InvScal(
         ConjType conjalpha,
         doff_t diagoffa,
         UpLo uploa,
         dim_t m,
         dim_t n,
         in ctype alpha,
         ref ctype a, inc_t rsa, inc_t csa
 );

    [DllImport(BlisDLL, EntryPoint = "bli_dscalm")]
    public static extern void Scal(
         ConjType conjalpha,
         doff_t diagoffa,
         UpLo uploa,
         dim_t m,
         dim_t n,
         in ctype alpha,
         ref ctype a, inc_t rsa, inc_t csa
 );

    [DllImport(BlisDLL, EntryPoint = "bli_dscal2m")]
    public static extern void Scal2(
         doff_t diagoffa,
         DiagType diaga,
         UpLo uploa,
         TransType transa,
         dim_t m,
         dim_t n,
         in ctype alpha,
         ref ctype a, inc_t rsa, inc_t csa,
         ref ctype b, inc_t rsb, inc_t csb
 );

    [DllImport(BlisDLL, EntryPoint = "bli_dsetm")]
    public static extern void Set(
         ConjType conjalpha,
         doff_t diagoffa,
         DiagType diaga,
         UpLo uploa,
         dim_t m,
         dim_t n,
         in ctype alpha,
         ref ctype a, inc_t rsa, inc_t csa
 );

    [DllImport(BlisDLL, EntryPoint = "bli_dsubm")]
    public static extern void Sub(
         doff_t diagoffa,
         DiagType diaga,
         UpLo uploa,
         TransType transa,
         dim_t m,
         dim_t n,
         ref ctype a, inc_t rsa, inc_t csa,
         ref ctype b, inc_t rsb, inc_t csb
 );
}
