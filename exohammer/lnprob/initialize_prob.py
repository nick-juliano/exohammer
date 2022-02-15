from exohammer.lnprob.lnprob_both import lnprob as both
from exohammer.lnprob.lnprob_ttv import lnprob as ttv
from exohammer.lnprob.lnprob_rv import lnprob as rv

def initialize_prob(system):
    if system.rvbjd == None and system.epoch != None:
        lnprob = ttv
    elif system.rvbjd != None and system.epoch == None:
        lnprob = rv
    elif system.rvbjd != None and system.epoch != None:
        lnprob = both
    
    return lnprob