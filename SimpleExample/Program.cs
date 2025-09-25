global using DDLA.Core;
using SimpleExample.LAFFExercise.LLS;
using SimpleExample.LAFFExercise.QRs;
using SimpleExample.SymmEVD;

namespace SimpleExample;

internal class Program
{
    static void Main(string[] args)
    {
        int cols = 977;
        int rows = cols + 4;
        //QRBase.TestQRFamily(rows, cols);
        //LLSBase.TestLLSFamily(rows, cols);
        TestSEvd.Test(cols);
    }
}
