import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import torch
import scipy.stats
import pandas as pd
from matplotlib.ticker import NullFormatter
import pathlib
import argparse

#sns.set_theme()
sns.set_context("paper",)
sns.set_style("ticks")
sns.set_style("whitegrid",rc={"grid.linewidth": 1})

#-------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--path', metavar='DIR', default = './data')

args = parser.parse_args()

try:
    acc_train = np.load(f'{args.path}/results/acc_chunk_c10_train_wr28_8.npy')
    acc_test = np.load(f'{args.path}/results/acc_chunk_c10_test_wr28_8.npy')
    r2 = np.load(f'{args.path}/results/r2_c10_test_wr28_8.npy')
    mean_corr = np.load(f'{args.path}/results/mean_corr_c10_test_wr28_8.npy')
    path = f'{args.path}/download_repr'
    ens_av = np.load(f'{path}/ensamble_averages_wide_resnet28_512_ep200_wd0.0005_label_smoothing0.0_n20_20_nrep10.npy')[0]

except:
    print('data not found. Run the "analysis_repr.py" script before')
    raise


#*******************************************************************************
fig = plt.figure(figsize = (9.5, 3))

gs = GridSpec(1, 1)
ax = fig.add_subplot(gs[0])
start = 4
chunk_size = acc_test[0]
sns.lineplot(ax =ax, x = chunk_size[start:], y=ens_av-acc_test[1][start:], marker = 'o', label = 'chunk', markersize = 6, color = 'C1')
yticks = np.array([0.3, 0.1, 0.03, 0.01, 0.003])
ax.plot(np.array(chunk_size[start:]), (0.12*np.array(chunk_size[start:])**-0.5), color = 'black',
                            linestyle = 'dashdot', label = '$w^{-0.5}$', alpha = 0.7, linewidth = '1.2')


ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xticks(chunk_size[start:])
ax.set_xticklabels(chunk_size[start:].astype(int))


ax.set_xlim(12, 600)
ax.axvspan(128, 3000, color = 'C1', alpha = 0.1)
ax.axvspan(64, 3000, color = 'C1', alpha = 0.1)
ax.set_xlabel('full net/chunk size', fontsize = 12)
ax.set_ylabel('error - error$_{∞}$', fontsize = 12)
ax.legend(loc = 'upper right', fontsize = '9')


lims = ax.get_ylim()
ax2 = ax.twinx()
ax2.set_yscale('log')
ax2.set_ylim(lims)

ax.set_yticks(yticks)
ax.set_yticklabels(yticks)
ax2.set_yticks(yticks)
labels = 1-ens_av + yticks
labels = np.around(labels, decimals=2)
ax2.set_yticklabels(labels)

ax.set_ylim(0.003, 0.5)
ax2.set_ylim(0.003, 0.5)
ax2.set_ylabel('test error', fontsize = 12)
ax.set_title('Test error chunks', fontsize = 14)
gs.tight_layout(fig, rect= [0.0, 0., 0.35, 0.96])





gs = GridSpec(1, 1)
ax = fig.add_subplot(gs[0])
sns.lineplot(ax =ax, x = chunk_size, y=1-acc_train[1], marker = 'o', label = 'chunk', markersize = 6, color = 'C1')

ax.set_xscale('log')
ax.set_xticks([2**i for i in range(1, 10, 2)])
ax.set_xticklabels([2**i for i in range(1, 10, 2)])

ax.set_yscale('symlog', linthresh = 10**-2)
ax.set_yticks([0.001, 0.01, 0.1, 1])
ax.set_ylim(-10**-3, 2)


ax.set_title('Training error chunks', fontsize = 14)
ax.set_xlabel('network size', fontsize = 12)
ax.set_ylabel('training error', fontsize = 11)

ax.set_xlim(0.8, 600)
ax.axvspan(128, 3000, color = 'C1', alpha = 0.1)
ax.axvspan(64, 3000, color = 'C1', alpha = 0.1)
gs.tight_layout(fig, rect= [0.37, 0., 0.65, 0.96])





#-------------------------------------------------------------------------------
yticks_err = [0, 0.25, 0.5, 0.75, 1]
yticks_r2 = [0,  0.5, 1]
wr2 = np.array([2**i for i in range(1, 10, 2)])


gs2 = GridSpec(1,1)
ax = fig.add_subplot(gs2[0])

#sns.lineplot(ax = ax, data = df_r2, x = 'width', y = 'r2', marker = 'o', markersize = 6, color = 'C0', label ='$R^2$')
sns.lineplot(ax = ax, x = r2[0], y = r2[1], marker = 'o', markersize = 6, color = 'C0', label ='$R^2$')
ax.legend(frameon=False, loc='center left', bbox_to_anchor=(0.42,0.57), fontsize = 10)
ax.set_ylabel('$R^2$', fontsize = 12)



ax.set_xlim(0.8, 600)
ax.axvspan(128, 3000, color = 'C1', alpha = 0.1)
ax.axvspan(64, 3000, color = 'C1', alpha = 0.1)

ax.set_title('', fontsize = 15)

ax2 = ax.twinx()
sns.lineplot(ax = ax2, x = mean_corr[0], y = mean_corr[1], marker = 'o',  markersize = 6, color = 'C2', label = 'residuals')
ax2.set_ylabel('')
ax2.legend(frameon=False, loc='center left', bbox_to_anchor=(0.42,0.43), fontsize = 10)
ax2.set_xscale('log')


ax2.yaxis.grid(False)
ax2.set_ylabel('mean corr.', fontsize = 12)
ax.set_title('$R^2$ - Mean corr. ', fontsize = 14)
ax2.ticklabel_format(axis="y", style="sci", scilimits = (0, 0))
ax.set_xlabel('chunk size', fontsize = 12)
ax.set_xticks(wr2)
ax.set_xticklabels(wr2)
gs2.tight_layout(fig, rect = [0.67, 0., 1, 0.97])

pathlib.Path('./plots').mkdir(parents=True, exist_ok=True)
plt.savefig('./plots/cifar10_wr28_plots.pdf', dpi = 200)


#*******************************************************************************

# cpt = [10, 20, 40, 60, 80, 100, 130, 150, 180, 200]
# gs = GridSpec(1, 1)
# ax = fig.add_subplot(gs[0])
# chunk_size = np.array([7, 15, 30, 60, 120, 240, 480]).astype('str')
# chunk_size
# for i in range(len(chunk_size)):
#     sns.lineplot(ax =ax, data = dyn_corr[i], x = 'epoch', y='corr', marker = 'o', label = chunk_size[i], markersize = 6)
# ax.set_xticks(cpt)
# ax.set_xticklabels(cpt)
# #ax.set_title('Correlation chunks', fontsize = 14)
# ax.set_xlabel('epoch', fontsize = 12)
# ax.set_ylabel('mean corr.', fontsize = 12)
# ax.set_xlim(0, 205)
# ax.legend(loc = 'upper right', title = 'chunk size', fontsize = '8')
# gs.tight_layout(fig, rect= [0.0, 0.05, 0.28, 0.48])
#
# #----------------------------------------------------------------------------
#
# cpt = [100, 130, 140, 150, 160, 180, 200]
# gs = GridSpec(1, 1)
# ax = fig.add_subplot(gs[0])
# chunk_size = np.array([88, 176, 352, 704, 1048])
# for i in range(1, len(chunk_size)):
#     sns.lineplot(ax =ax, data = dyn_chunks[i], x = 'epoch', y='error', marker = 'o', label = chunk_size[i], markersize = 6)
# ax.set_xticks(cpt)
# ax.set_xticklabels(cpt)
# #ax.set_title('Dynamics chunks', fontsize = 14)
# ax.set_xlabel('epoch', fontsize = 12)
# ax.set_ylabel('error', fontsize = 12)
# ax.set_xlim(130, 205)
# #ax.set_ylim(0, 0.35)
# ax.legend(loc = 'upper right', title = 'chunk size', fontsize = '9')
# gs.tight_layout(fig, rect= [0.34, 0.05, 0.64, 0.48])
#
# #-------------------------------------------------------------------------------
# gs = GridSpec(1, 1)
# ax = fig.add_subplot(gs[0])
# sns.lineplot(ax =ax, data = 1-acc_train, label = 'training error')
# sns.lineplot(ax =ax, data = 1-acc_test, label = 'test error')
# ax.set_xticks([150, 160, 170, 180, 190, 200])
# ax.set_xticklabels([150, 160, 170, 180, 190, 200])
# #ax.set_title('Training and test error', fontsize = 14)
# ax.set_xlabel('epoch', fontsize = 12)
# ax.set_ylabel('error', fontsize = 12)
# ax.set_ylim(-0.005, 0.09)
# ax.set_xlim(150, 202)
# ax.axvline(170, linestyle = '--', label = 'end of phase 1')
# ax.axvline(180, linestyle = 'dashdot', label = 'end of phase 2')
# ax.legend(fontsize = '8')
# gs.tight_layout(fig, rect= [0.69, 0.05, 0.98, 0.48])


fig.text(0.04, 0.92, 'a', weight = 'bold', fontsize = 14)
fig.text(0.38, 0.92, 'b', weight = 'bold', fontsize = 14)
fig.text(0.7, 0.92, 'c', weight = 'bold', fontsize = 14)
# fig.text(0.04, 0.47, 'd', weight = 'bold', fontsize = 14)
# fig.text(0.38, 0.47, 'e', weight = 'bold', fontsize = 14)
# fig.text(0.7, 0.47, 'f', weight = 'bold', fontsize = 14)
