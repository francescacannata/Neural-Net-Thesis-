"""----------------------------------------------------------------------
 File Name: plot_comparison.py
 Goal: Plot piecewise approximation errors for different decay types
 Author: Francesca Cannata
--------------------------------------------------------------------------"""
import pandas as pd
import matplotlib.pyplot as plt

# Load CSVs
df_pol  = pd.read_csv('PwErrors_pol.csv')
df_exp  = pd.read_csv('PwErrors_exp.csv')
#df_none = pd.read_csv('PwErrors_noDecay.csv')

plt.figure(figsize=(10, 8))

# Experimental errors
plt.loglog(df_pol['k'],  df_pol['error'],  color='C1', label='Experimental error (polynomial decay)', linewidth=2.3)
plt.loglog(df_exp['k'],  df_exp['error'],  color='orange', label='Experimental error (exponential decay)', linewidth=2.3)
#plt.loglog(df_none['k'], df_none['error'], color='C4', label='Experimental Error (no decay)',           linewidth=2.3)

# Theoretical error (from polynomial run)
plt.loglog(df_pol['k'], df_pol['error_th'], color='purple',
           label='Theoretical error (polynomial decay)', linewidth=2.3)

# Theoretical error (from exponential run, as reference)
plt.loglog(df_exp['k'], df_exp['error_th'], color='C4',
           label='Theoretical error (exponential decay)', linewidth=2.3)

plt.xlabel('Intervals $k$', fontsize=16)
plt.ylabel('MSE loss', fontsize=16)
plt.tick_params(axis='both', labelsize=14)
plt.legend(loc='best', fontsize=14)
plt.grid(True, which='major', linewidth=0.5)
plt.grid(True, which='minor', linewidth=0.3, linestyle=':')
plt.savefig('piecewise_combined.pdf', bbox_inches='tight', dpi=300)
plt.show()