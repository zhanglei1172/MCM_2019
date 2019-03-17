#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
PATH = '/home/zhang/下载/2019_MCM-ICM_Problems/2018_MCMProblemC_DATA/'
MCM_NFLIS_Data = pd.read_excel(PATH + 'MCM_NFLIS_Data.xlsx', sheet_name='Data')
import plotly.plotly as py
import plotly.figure_factory as ff
from plotly import tools
tools.set_credentials_file(username='MathCoder_', api_key='3Dvx3VhwfiKCkyYxxJDl')

fips = np.unique(MCM_NFLIS_Data.FIPS_Combined.values).tolist()
values = range(len(fips))

fig = ff.create_choropleth(fips=fips, values=values)
py.iplot(fig, filename='choropleth of some cali counties - full usa scope')
plt.show()