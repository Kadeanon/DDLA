global using DDLA.Core;
using SimpleExample.SVD;
using SimpleExample.SymmEVD;

namespace SimpleExample;

internal class Program
{
    static void Main(string[] args)
    {
        int cols = 1023;
        int rows = cols;
        //QRBase.TestQRFamily(rows, cols);
        //LLSBase.TestLLSFamily(rows, cols);
        //TestSEvd.Test(cols);
        TestSvd.Test(rows, cols);
    }
}
