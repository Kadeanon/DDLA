namespace SimpleExample.LAFFExercise.QR;

/// <summary>
/// Blocked Householder QR factorization
/// with UT representation
/// </summary>
public class HHUTQR(Matrix A) : QRBase(A)
{
    public override bool IsEconomy => true;

    public const int MaxBlockSize = 128;

    public int BlockSize { get; }
        = Math.Min(MaxBlockSize, A.Cols);

    public override void Kernel()
    {
        throw new NotImplementedException();
    }
}
