import statsmodels.formula.api as smf


def mixed_effect_model(dataframe, dependent, independent):
    """A mixed effect linear model.

    Parameters
    ----------
    dataframe : dataframe
        A pandas dataframe containing the dependent and independent data
        with group data (usually this is the subject).
    dependent : str
        A string stating which variable to use as dependent variable
    independent : str
        A string stating which variable to use as independent variable


    Returns
    -------
    stat model
        A fitted mixed effect model

    """

    md = smf.mixedlm(dependent + '~' + independent,
                     dataframe,
                     groups=dataframe["subject"])
    mdf = md.fit(disp=False)
    print(mdf.summary())

    return mdf
