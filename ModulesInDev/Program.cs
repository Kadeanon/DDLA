global using DDLA.Core;
using ModulesInDev.SymmEVD;

namespace ModulesInDev;

internal class Program
{
    static void Main(string[] args)
    {
        int cols = 1025;
        int rows = cols + 3;
        //QRBase.TestQRFamily(rows, cols);
        //LLSBase.TestLLSFamily(rows, cols);
        TestSEvd.Test(rows);
        //TestSvd.Test(rows, cols);
    }
}
