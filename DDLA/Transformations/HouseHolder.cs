// These algorithms are ported from LibFlame.
// https://github.com/flame/libflame

using DDLA.BLAS;
using DDLA.Core;
using DDLA.Misc.Flags;

namespace DDLA.Transformations;

public static class HouseHolder
{

    /// <summary>
    /// Builds a HouseHolder transformation matrix H, 
    /// such that when applied to a vector x, 
    /// it transforms it into a vector with 
    /// the first element equal to chi and the rest zero.
    /// </summary>
    /// <param name="chi">The first element of vector x to be projected. 
    /// When back, it has been override with the projected value rho.
    /// </param>
    /// <param name="xLast">The last part of vector x to be projected. 
    /// When back, it has been override with the last part of u.</param>
    /// <param name="tau">The factor to scalar the matrix. 
    /// H = (I - inv(<paramref name="tau"/>)uu^T</param>
    /// <remarks>
    /// The H is built implicitly as UT used by libflame,
    /// where u2 has implicitly first element 1, other elements
    /// in <paramref name="xLast"/>, 
    /// and a scalar factor <paramref name="tau"/>
    /// to inv-scaling the outer matrix.
    /// </remarks>
    public static void BuildHouseHolder(ref double chi, VectorView xLast, out double tau)
    {
        double lenLast = BlasProvider.NrmF(xLast);

        if (lenLast == 0.0)
        {
            // for lenlast = 0, we can write reflection directly.
            chi = -chi;
            tau = 0.5;
            return;
        }

        double alpha = double.Hypot(chi, lenLast);

        double rho = -double.CopySign(alpha, chi);
        double miu = chi - rho;
        chi = rho;

        xLast.InvScaled(miu);
        double scaledLenLast = lenLast / Math.Abs(miu);

        tau = (1 + scaledLenLast * scaledLenLast) / 2;
    }

    /// <summary>
    /// Apply a HouseHolder transformation H to a matrix A.
    /// This method applies the implicitly stored Householder H with
    /// vector u(<paramref name="u2"/> with implicitly first element 1) 
    /// and scalar (<paramref name="tau"/>) to the target matrix. 
    /// The target matrix has two part: the top vector <paramref name="a1"/> 
    /// and bottom submatrix <paramref name="A2"/>.
    /// The transformation is performed in-place, overwriting the original data
    /// in <paramref name="a1"/> and <paramref name="A2"/>.
    /// </summary>
    /// <param name="side">Specifies whether 
    /// the transformation is applied from the left or right.</param>
    /// <param name="tau">The scalar factor for the Householder transformation.
    /// H = (I - inv(<paramref name="tau"/>)uu^T</param>
    /// <param name="u2">The last part of u.</param>
    /// <param name="a1">The top part of A.</param>
    /// <param name="A2">The bottom part of A.</param>
    internal static void ApplyHouseHolder(SideType side,
        ref double tau, VectorView u2, VectorView a1, MatrixView A2)
    {
        if (a1.Length == 0 || tau == 0.0)
            return;
        var w = Vector.Create(a1.Length);
        a1.CopyTo(w);

        if (side == SideType.Left)
        {
            //BlasProvider.GeMV(1, A2.T, u2, 1, w);
            //BlasProvider.InvScal(tau, w);
            //BlasProvider.Axpy(-1, w, a1);
            //BlasProvider.GeR(-1, u2, w, A2);
            u2.LeftMul(A2, 1.0, w);
            w.InvScaled(tau);
            a1.Subtracted(w);
            A2.Rank1(-1.0, u2, w);

        }
        else if (side == SideType.Right)
        {
            //BlasProvider.GeMV(1, A2, u2, 1, w);
            //BlasProvider.InvScal(tau, w);
            //BlasProvider.Axpy(-1, w, a1);
            //BlasProvider.GeR(-1, w, u2, A2);
            A2.Multify(u2, 1.0, w);
            w.InvScaled(tau);
            a1.Subtracted(w);
            A2.Rank1(-1.0, w, u2);
        }
    }
}
