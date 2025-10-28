using System.Runtime.InteropServices;

using ctype = double;
using rtype = double;
using DDLA.Misc.Flags;

namespace DDLA.BLAS;

public static partial class Source
{

    [DllImport(BlisDLL, EntryPoint = "bli_dgemm")]
    public static extern void GeMM(
         TransType transa,
         TransType transb,
         dim_t m,
         dim_t n,
         dim_t k,
         in ctype alpha,
         ref ctype a, inc_t rsa, inc_t csa,
         ref ctype b, inc_t rsb, inc_t csb,
         in ctype beta,
         ref ctype c, inc_t rsc, inc_t csc
 );

    [DllImport(BlisDLL, EntryPoint = "bli_dgemmt")]
    public static extern void GeMMT(
         UpLo uploc,
         TransType transa,
         TransType transb,
         dim_t m,
         dim_t k,
         in ctype alpha,
         ref ctype a, inc_t rsa, inc_t csa,
         ref ctype b, inc_t rsb, inc_t csb,
         in ctype beta,
         ref ctype c, inc_t rsc, inc_t csc
 );

    [DllImport(BlisDLL, EntryPoint = "bli_dhemm")]
    public static extern void HeMM(
         SideType sidea,
         UpLo uploa,
         ConjType conja,
         TransType transb,
         dim_t m,
         dim_t n,
         in ctype alpha,
         ref ctype a, inc_t rsa, inc_t csa,
         ref ctype b, inc_t rsb, inc_t csb,
         in ctype beta,
         ref ctype c, inc_t rsc, inc_t csc
 );

    [DllImport(BlisDLL, EntryPoint = "bli_dherk")]
    public static extern void HeRk(
         UpLo uploc,
         TransType transa,
         dim_t m,
         dim_t k,
         ref rtype alpha,
         ref ctype a, inc_t rsa, inc_t csa,
         ref rtype beta,
         ref ctype c, inc_t rsc, inc_t csc
 );

    [DllImport(BlisDLL, EntryPoint = "bli_dher2k")]
    public static extern void HeR2k(
         UpLo uploc,
         TransType transa,
         TransType transb,
         dim_t m,
         dim_t k,
         in ctype alpha,
         ref ctype a, inc_t rsa, inc_t csa,
         ref ctype b, inc_t rsb, inc_t csb,
         ref rtype beta,
         ref ctype c, inc_t rsc, inc_t csc
 );

    [DllImport(BlisDLL, EntryPoint = "bli_dsymm")]
    public static extern void SyMM(
         SideType sidea,
         UpLo uploa,
         ConjType conja,
         TransType transb,
         dim_t m,
         dim_t n,
         in ctype alpha,
         ref ctype a, inc_t rsa, inc_t csa,
         ref ctype b, inc_t rsb, inc_t csb,
         in ctype beta,
         ref ctype c, inc_t rsc, inc_t csc
 );

    [DllImport(BlisDLL, EntryPoint = "bli_dsyrk")]
    public static extern void SyRk(
         UpLo uploc,
         TransType transa,
         dim_t m,
         dim_t k,
         in ctype alpha,
         ref ctype a, inc_t rsa, inc_t csa,
         in ctype beta,
         ref ctype c, inc_t rsc, inc_t csc
 );

    [DllImport(BlisDLL, EntryPoint = "bli_dsyr2k")]
    public static extern void SyR2k(
         UpLo uploc,
         TransType transa,
         TransType transb,
         dim_t m,
         dim_t k,
         in ctype alpha,
         ref ctype a, inc_t rsa, inc_t csa,
         ref ctype b, inc_t rsb, inc_t csb,
         in ctype beta,
         ref ctype c, inc_t rsc, inc_t csc
 );

    [DllImport(BlisDLL, EntryPoint = "bli_dtrmm")]
    public static extern void TrMM(
         SideType sidea,
         UpLo uploa,
         TransType transa,
         DiagType diaga,
         dim_t m,
         dim_t n,
         in ctype alpha,
         ref ctype a, inc_t rsa, inc_t csa,
         ref ctype b, inc_t rsb, inc_t csb
 );

    [DllImport(BlisDLL, EntryPoint = "bli_dtrmm3")]
    public static extern void TrMM3(
         SideType sidea,
         UpLo uploa,
         TransType transa,
         DiagType diaga,
         TransType transb,
         dim_t m,
         dim_t n,
         in ctype alpha,
         ref ctype a, inc_t rsa, inc_t csa,
         ref ctype b, inc_t rsb, inc_t csb,
         in ctype beta,
         ref ctype c, inc_t rsc, inc_t csc
 );

    [DllImport(BlisDLL, EntryPoint = "bli_dtrsm")]
    public static extern void TrSM(
         SideType sidea,
         UpLo uploa,
         TransType transa,
         DiagType diaga,
         dim_t m,
         dim_t n,
         in ctype alpha,
         ref ctype a, inc_t rsa, inc_t csa,
         ref ctype b, inc_t rsb, inc_t csb
 );
}
