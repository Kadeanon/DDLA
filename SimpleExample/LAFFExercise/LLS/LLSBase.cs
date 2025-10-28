namespace SimpleExample.LAFFExercise.LLS;

public abstract class LLSBase
{
    public MatrixView A { get; }

    public VectorView b { get; }

    public LLSBase(MatrixView A, VectorView b)
    {
        this.A = A;
        this.b = b;
    }

    public abstract Vector Kernel();

    public static void TestLLSFamily(int rows, int cols)
    {
        var A = Matrix.RandomDense(rows, cols);
        var b = Vector.Random(rows);

        Console.WriteLine("~~~~~~~~Normal Equation~~~~~~~~");
        var ne = new NELLS(A, b);
        TestLLS(ne);
        Console.WriteLine();
        Console.WriteLine();

        Console.WriteLine("~~~~~~~~Modified Gram-Schmidt~~~~~~~~");
        var qr = new QRLLS(A, b);
        TestLLS(qr);
        Console.WriteLine();
        Console.WriteLine();
    }

    public static void TestLLS(LLSBase lls)
    {
        var A = lls.A;
        var b = lls.b;
        //Console.WriteLine($"A:\n{A}");
        //Console.WriteLine($"b:\n{b}");
        var x = lls.Kernel();
        //Console.WriteLine($"x:\n{x}");
        //Console.WriteLine($"R:\n{R}");
        var build = A * x;
        //Console.WriteLine($"Ax:\n{build}");
        var diff = b - build;
        Console.WriteLine($"nrm2(b - Ax):" +
            $"{diff.NrmF()}");
    }
}
