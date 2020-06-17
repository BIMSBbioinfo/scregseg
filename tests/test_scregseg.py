from scregseg import DirMulHMM
from scregseg import Scregseg

def test_hmm():
    Scregseg(DirMulHMM(20))
