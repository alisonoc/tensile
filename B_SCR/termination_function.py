import pandas as pd
from B_SCR.Acquisition_utilies import AcquisitionUtilities


def termination_check(i, bayes_dic, utility):
    """ FUNCTION TO DEFINE WHETHER SIMULATION SHOULD
    EXIT PRIOR TO MAXIMUM NUMBER OF ITERATIONS
     INPUTS:
     I: JOB ITERATOR NUMBER, COUNTING NUMBER OF MAIN JOBS (I.E. AFTER INTITAL JOBS)
     BAYES_DIC: DICTIONARY OF RESULTS FOR ALL ITERATIONS TO DATE
     UTILITY: CURRENT AQUISITION FUNCTION IN USE"""
    terminate = False
    # ##CONVERT BAYES TO DF
    df = pd.DataFrame.from_dict(bayes_dic)
    # ##MINIMUM MAPE
    min_mape = df['MAPE'].sort_values(ascending=True).iloc[0]
    # ##PREVIOUS THREE KAPPA VALUES
    three_kappa = df['KAPPA'].iloc[-4:-1]
    if min_mape <= 3.0:
        terminate = True
    # ##KAPPA ZERO
    elif (three_kappa.all()==0) or (i % 20 == 0):
        # ##EXPLORE
        ckappa = 2.5
    # ##GET THE LAST KNOWN MAPE AND ROC IF BOTH ARE REDUCING THEN REDUCE KAPPA
    elif (df['ROC_BEGIN'].iloc[-1] < 0) and (df['ROC_THREE'].iloc[-1] < 0):
        ckappa = max(0, utility.kappa - 0.5) # ##KAPPA CAN'T BE LESS THAN ZERO
    else:
        ckappa = df['KAPPA'].min()
    # ##SET UTILITY WITH NEW KAPPA
    try:
        utility = AcquisitionUtilities(aq_type="UCB", kappa=ckappa, xi=0)
    except:
        utility = AcquisitionUtilities(aq_type="UCB", kappa=2.5, xi=0)

    return terminate, bayes_dic, utility
