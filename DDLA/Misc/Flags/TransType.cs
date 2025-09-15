namespace DDLA.Misc.Flags;

public enum TransType
{
    NoTrans = 0b00 << 3,
    OnlyTrans = 0b01 << 3,
    OnlyConj = 0b10 << 3,
    ConjTrans = 0b11 << 3
}
