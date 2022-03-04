from exohammer.lnprob.lnprob_both import lnprob as both
from exohammer.lnprob.lnprob_ttv import lnprob as ttv
from exohammer.lnprob.lnprob_rv import lnprob as rv


def initialize_prob(System):
    if System.rvbjd is None and System.epoch is not None:
        lnprob = ttv
    elif System.rvbjd is not None and System.epoch is None:
        lnprob = rv
    else:
        lnprob = both

    return lnprob
