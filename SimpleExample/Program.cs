global using DDLA.Core;
using SimpleExample.LAFFExercise.QR;

namespace SimpleExample;

internal class Program
{
    static void Main(string[] args)
    {
        int rows = 10;
        int cols = rows;
        TestQRFamily(rows, cols);
    }

    static void TestQRFamily(int rows, int cols)
    {
        var A = Matrix.RandomDense(rows, cols);
        Console.WriteLine("~~~~~~~~Classic Gram-Schmidt~~~~~~~~");
        var cgs = new CGSQR(A);
        TestQR(cgs);
        Console.WriteLine();
        Console.WriteLine();

        Console.WriteLine("~~~~~~~~Modified Gram-Schmidt~~~~~~~~");
        var mgs = new MGSQR(A);
        TestQR(mgs);
        Console.WriteLine();
        Console.WriteLine();

        Console.WriteLine("~~~~~~~~Householder~~~~~~~~");
        var hh = new HHQR(A);
        TestQR(hh);
        Console.WriteLine();
        Console.WriteLine();

        //Console.WriteLine("~~~~~~~~Householder WY~~~~~~~~");
        //var hhwy = new HHWYQR(A);
        //TestQR(hhwy);
        //Console.WriteLine();
        //Console.WriteLine();
    }

    static void TestQR(QRBase qr)
    {
        var A = qr.Orig;
        //Console.WriteLine($"Original A:\n{A}");
        qr.Kernel();
        var Q = qr.Q;
        var R = qr.R;
        //Console.WriteLine($"Q:\n{Q}");
        //Console.WriteLine($"R:\n{R}");
        var A_reconstructed = Q * R;
        //Console.WriteLine($"Q*Q^T:\n" +
        //    $"{Q * Q.Transpose()}");
        //Console.WriteLine("A reconstructed(Q*R):\n" +
        //    $"{A_reconstructed}");
        var diff = A_reconstructed - A;
        Console.WriteLine($"nrm2(diff):" +
            $"{diff.NrmF()}");
    }
}
