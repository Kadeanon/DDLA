using System.Diagnostics.CodeAnalysis;

namespace SimpleExample.LAFFExercise.QR;

public abstract class QRBase
{
    public abstract bool IsEconomy { get; }

    public Matrix Orig { get; }

    public Matrix A { get; }

    public Matrix? Q { get; protected set; }

    public Matrix? R { get; protected set; }

    public QRBase(Matrix A)
    {
        if (A.Rows < A.Cols)
        {
            throw new ArgumentException
            ("Rows must be no less than Cols");
        }
        Orig = A;
        this.A = A.Clone();
    }

    [MemberNotNull(nameof(R), nameof(Q))]
    public abstract void Kernel();
}
