using ctype = double;
using rtype = double;
using System.Runtime.InteropServices;
using DDLA.Misc.Flags;

namespace DDLA.BLAS;

public static partial class Source
{
    [DllImport(BlisDLL, EntryPoint = "bli_daddd")]
    public static extern void Add(
         doff_t diagoffa,
         DiagType diaga,
         TransType transa,
         dim_t m,
         dim_t n,
         ref ctype a, inc_t rsa, inc_t csa,
         ref ctype b, inc_t rsb, inc_t csb
 );

    [DllImport(BlisDLL, EntryPoint = "bli_daxpyd")]
    public static extern void Axpy(
         doff_t diagoffa,
         DiagType diaga,
         TransType transa,
         dim_t m,
         dim_t n,
         in ctype alpha,
         ref ctype a, inc_t rsa, inc_t csa,
         ref ctype b, inc_t rsb, inc_t csb
 );

    [DllImport(BlisDLL, EntryPoint = "bli_dcopyd")]
    public static extern void Copy(
         doff_t diagoffa,
         DiagType diaga,
         TransType transa,
         dim_t m,
         dim_t n,
         ref ctype a, inc_t rsa, inc_t csa,
         ref ctype b, inc_t rsb, inc_t csb
 );

    [DllImport(BlisDLL, EntryPoint = "bli_dinvertd")]
    public static extern void Invert(
         doff_t diagoffa,
         dim_t m,
         dim_t n,
         ref ctype a, inc_t rsa, inc_t csa
 );

    [DllImport(BlisDLL, EntryPoint = "bli_dinvscald")]
    public static extern void InvScal(
         ConjType conjalpha,
         doff_t diagoffa,
         dim_t m,
         dim_t n,
         in ctype alpha,
         ref ctype a, inc_t rsa, inc_t csa
 );

    [DllImport(BlisDLL, EntryPoint = "bli_dscald")]
    public static extern void Scal(
         ConjType conjalpha,
         doff_t diagoffa,
         dim_t m,
         dim_t n,
         in ctype alpha,
         ref ctype a, inc_t rsa, inc_t csa
 );

    [DllImport(BlisDLL, EntryPoint = "bli_dscal2d")]
    public static extern void Scal2(
         doff_t diagoffa,
         DiagType diaga,
         TransType transa,
         dim_t m,
         dim_t n,
         in ctype alpha,
         ref ctype a, inc_t rsa, inc_t csa,
         ref ctype b, inc_t rsb, inc_t csb
 );

    [DllImport(BlisDLL, EntryPoint = "bli_dsetd")]
    public static extern void Set(
         ConjType conjalpha,
         doff_t diagoffa,
         dim_t m,
         dim_t n,
         in ctype alpha,
         ref ctype a, inc_t rsa, inc_t csa
 );

    [DllImport(BlisDLL, EntryPoint = "bli_dshiftd")]
    public static extern void Shift(
         doff_t diagoffa,
         dim_t m,
         dim_t n,
         in ctype alpha,
         ref ctype a, inc_t rsa, inc_t csa
 );

    [DllImport(BlisDLL, EntryPoint = "bli_dsubd")]
    public static extern void Sub(
         doff_t diagoffa,
         DiagType diaga,
         TransType transa,
         dim_t m,
         dim_t n,
         ref ctype a, inc_t rsa, inc_t csa,
         ref ctype b, inc_t rsb, inc_t csb
 );

    [DllImport(BlisDLL, EntryPoint = "bli_dxpbyd")]
    public static extern void Xpby(
         doff_t diagoffa,
         DiagType diaga,
         TransType transa,
         dim_t m,
         dim_t n,
         ref ctype a, inc_t rsa, inc_t csa,
         in ctype beta,
         ref ctype b, inc_t rsb, inc_t csb
 );
}
