using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;

namespace DDLA.Core;

public class LinalgException : Exception
{
    internal string FunctionName { get; }

    internal int ErrorCode { get; }

    public LinalgException(string functionName,
        int errorCode, string message) : base(
            $"MKL function {functionName} failed " +
        $"with errorCode {errorCode}: {message}")
    {
        FunctionName = functionName;
        ErrorCode = errorCode;
    }

    public LinalgException(string functionName, string message)
        : base($"Managed Linalg function {functionName} " +
              $"failed: {message}")
    {
        FunctionName = functionName;
        ErrorCode = 0;
    }

    [DebuggerHidden]
    [DoesNotReturn]
    public static void ThrowConvergenceFailed(string functionName = "unknown")
    {
        throw new LinalgException(functionName,
            $"convergence failed for {functionName}.");
    }
}
