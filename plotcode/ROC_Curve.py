import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Loading data
path_train = os.getcwd() + '/plotcode/ROC_train.csv'
path_val = os.getcwd() + '/plotcode/ROC_val.csv'

df_train = pd.read_csv(path_train)
df_val = pd.read_csv(path_val)

# Plotting
fig, ax= plt.subplots()

ax.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r', label='Chance') # Baseplot with red chance line


roc_auc = auc(df_train['fpr_train'], df_train['tpr_train'])
ax.plot(df_train['fpr_train'], df_train['tpr_train'], color='black', linestyle='dashed', label='Training ROC (AUROC: {:.4f})'.format(roc_auc))



roc_auc = auc(df_val['fpr_val'], df_val['tpr_val'])
ax.plot(df_val['fpr_val'], df_val['tpr_val'], color='black', label='Validation ROC (AUROC: {:.4f})'.format(roc_auc), lw=2)

ax.set_xlim([-0.05, 1.05])
ax.set_ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
# plt.title('Receiver operating characteristic curve')

plt.legend(loc="lower right")
# fig.legend(loc=1)


plt.savefig(os.getcwd() + '/plotcode/' + 'i3dautoscore_ROCcurve.pdf')
