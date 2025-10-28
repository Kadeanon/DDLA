using DDLA.Factorizations;

namespace SimpleExample.LAFFExercise.LLS;

public class QRLLS(MatrixView A, VectorView b) : LLSBase(A, b)
{
    public override Vector Kernel()
    {
        var qr = new QR(A);
        return qr.Solve(b);
    }
}
