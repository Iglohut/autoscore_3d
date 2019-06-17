import pandas as pd
import os
import matplotlib.pyplot as plt

path_csv = os.getcwd() + '/project/henk_metrics_logger.csv'
df = pd.read_csv(path_csv)

# Initiate figure
fig, ax1 = plt.subplots()

# Left side
color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('AUROC', color=color)
ax1.plot(df['Epoch'], df['val auc'], color=color)
ax1.plot(df["Epoch"], df['train auc'], color=color, linestyle='dashed')
ax1.tick_params(axis='y', labelcolor=color)
# plt.legend(['Validation AUC', 'Training AUC'], loc="center")

# Right side
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Loss', color=color)  # we already handled the x-label with ax1
ax2.plot(df['Epoch'], df['val_loss'], color=color)
ax2.plot(df["Epoch"], df['train_loss'], color=color, linestyle='dashed')
ax2.tick_params(axis='y', labelcolor=color)
# plt.legend(['Validation loss', 'Training loss'], loc="center right")


leg = fig.legend(['Validation AUROC', 'Training AUROC', 'Validation loss', 'Training loss'], loc='center')
if leg:
    leg.draggable()

# fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.savefig(os.getcwd() + '/plotcode/' + 'i3dautoscore_Aucloss.pdf')

