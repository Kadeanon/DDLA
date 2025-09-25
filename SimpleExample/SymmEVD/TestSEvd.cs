using DDLA.BLAS;
using SimpleExample.SymmEVD.Diag;
using SimpleExample.SymmEVD.Tradiag;
using System.Diagnostics;

namespace SimpleExample.SymmEVD
{
    internal class TestSEvd
    {
        internal static void Test()
        {
            for (int i = 0; i < 20; i++)
            {
                Console.WriteLine($"--- Iteration {i} ---");
                TestHHUubTridiag(1024);
                Console.WriteLine();
            }
        }

        static void TestHHUubTridiag(int len)
        {
            DateTime start = DateTime.Now;
            var mat = Matrix.RandomSPD(len);
            var orig = mat.Clone();

            TridiagBase tridiag = new HHUnbTradiag(mat);
            tridiag.Kernel();
            var Q = tridiag.Q;
            var trans = Q.EmptyLike().T;
            Q.CopyTo(trans);
            Q = trans;
            var T = tridiag.GetTriMatrix();

            Console.WriteLine($"nrmf(diff)={(Q * (T * Q.T) - mat).NrmF()}");
            Console.WriteLine($"nrmf(QQT-I)={(Q * Q.T - Matrix.Eyes(len)).NrmF()}");

            var span = DateTime.Now - start;
            Console.WriteLine($"Tridiag time out: {span}");

            start = DateTime.Now;
            var diag = tridiag.Diag;
            var fran = new FrancisQR(diag,
                tridiag.SubDiag, Q);
            fran.Kernel();
            var eigenValues = diag;

            span = DateTime.Now - start;
            Console.WriteLine($"Final diag time out: {span}");

            for (var i = 0; i < len; i++)
            {
                var eigenValue = eigenValues[i];
                var eigenVector = Q.GetColumn(i);
                var diffMax = (orig * eigenVector - eigenValue * eigenVector).MaxAbs();
                if (diffMax > 1e-12)
                {
                    Console.WriteLine($"[WARNING]Index {i}: eigenvalue {eigenValue}, max residual {diffMax}");
                }
            }
        }
    }
}
