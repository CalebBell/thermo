# -*- coding: utf-8 -*-
"""
test MOSCED
author: Edgar Sanchez
"""

from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from thermo.mosced import MOSCED
from thermo.mosced import mosced_params_2005
import os
 

### --- Get MOSCED predictions from DDBST http://ddbonline.ddbst.com/MOSCEDCalculation/MOSCEDCGI.exe
# Inchikeys and names are both in the same order

def get_gamma_MOSCED_DDB(solvent, solute, T):
    website_link = 'http://ddbonline.ddbst.com/MOSCEDCalculation/MOSCEDCGI.exe?component1=' \
        + solvent +'&component2='+ solute +'&temperatures=' + str(T)
    webpage = requests.get(website_link)
    soup    = BeautifulSoup(webpage.content, 'html.parser')
    table_info = soup.findAll(class_="tableformat_no_width")[1]
    df        = pd.read_html(str(table_info))[0]
    gamma = df.iloc[0,2]
    
    return gamma

if not os.path.isfile('Data/MOSCED_DDB.csv'):
    with open('Data/mosced_compounds.txt', 'r') as f:
        names = f.readlines()
        
    gammas_DDB = np.zeros((len(names), len(names)))
        
    for i, name in enumerate(names):
        print('-'*60)
        print('Solvent: ', i+1)
        if i == len(names)-1:
            solvent = name
        else:
            solvent = name[:-1]
        for j, name in enumerate(tqdm(names)):
            if j == len(names)-1:
                solute = name
            else:
                solute = name[:-1]
            if solvent == solute:
                gamma = 1.
            else:
                gamma = get_gamma_MOSCED_DDB(solvent, solute, T=298.15)
            gammas_DDB[i,j] = gamma
    df = pd.DataFrame(gammas_DDB)
    df.to_csv('Data/MOSCED_DDB.csv')
else:
    gammas_DDB = pd.read_csv('Data/MOSCED_DDB.csv', index_col=0).values

### --- Get MOSCED predictions with local implementation

gammas_local = np.zeros(gammas_DDB.shape)

binary_sys = MOSCED()
for i, solvent in enumerate(mosced_params_2005):
    solvent_params = binary_sys.get_mosced_params(solvent)
    for j, solute in enumerate(mosced_params_2005):
        solute_params = binary_sys.get_mosced_params(solute)
        binary_sys.load_mosced_parameters(298.15, solvent_params, solute_params)
        gammas_local[i,j] = binary_sys.gamma_infinite_dilution()
        
        error = gammas_DDB[i,j] - np.round(gammas_local[i,j],2)
        if error > 100:
            with open('Data/mosced_compounds.txt', 'r') as f:
                names = f.readlines()
            print('-'*50)
            print(i,j)
            print(gammas_DDB[i,j])
            print(np.round(gammas_local[i,j],2))
            print(error)
            print(solvent)
            print(solute)
            print(names[i])
            print(names[j])
            
        
df = pd.DataFrame(gammas_local)
df.to_csv('Data/MOSCED_local.csv')

error = gammas_DDB - np.round(gammas_local,2)
df = pd.DataFrame(error)
df.to_csv('Data/MOSCED_error.csv')

        
        
        








