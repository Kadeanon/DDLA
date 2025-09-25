using DDLA.Factorizations;

namespace SimpleExample.LAFFExercise.LLS
{
    public class NELLS(MatrixView A, VectorView b) : LLSBase(A, b)
    {
        public override Vector Kernel()
        {
            return new Cholesky(A.T * A, inplace: true).Solve(A.T * b);
        }
    }
}
