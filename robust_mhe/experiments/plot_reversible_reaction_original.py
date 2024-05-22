import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BETA = [0.0001]
font = 25
axes_pos = [0.25, 0.2, 0.65, 0.6]
matplotlib.rcParams['font.family'] = ['serif']

palette = sns.color_palette("tab10", len(BETA) + 2)

ukf_error, mhe_error, robust_mhes_error = np.load(
    '../results/reversible_reaction/original_0.2.pk', allow_pickle=True)
time_length = int(ukf_error.shape[1])

pd0_list, pd1_list = [], []
pd_list = [pd0_list, pd1_list]
for i in range(ukf_error.shape[2]):
    for j in range(ukf_error.shape[0]):
        pd_list[i].append(pd.DataFrame({'Error': ukf_error[j, 0:time_length, i],
                                        'Algorithm': 'UKF',
                                        'Step': np.arange(time_length)
                                        }))

        pd_list[i].append(pd.DataFrame({'Error': mhe_error[j, 0:time_length, i],
                                        'Algorithm': 'MHE',
                                        'Step': np.arange(time_length)}))

        for k, beta in enumerate(BETA):
            string = 'beta=' + str(beta)
            pd_list[i].append(pd.DataFrame({'Error': robust_mhes_error[j, k, 0:time_length, i],
                                            'Algorithm': string,
                                            'Step': np.arange(time_length)}))
pd_error0 = pd.concat(pd_list[0])
pd_error1 = pd.concat(pd_list[1])
pd_error0 = pd_error0.reset_index(drop=True)  # Apply reset_index function, drop=False时, 保留旧的索引为index列
pd_error1 = pd_error1.reset_index(drop=True)  # Apply reset_index function, drop=False时, 保留旧的索引为index列

f1 = plt.figure(1)
ax1 = f1.add_axes(axes_pos)
g1 = sns.lineplot(x='Step', y="Error", hue="Algorithm", style="Algorithm", data=pd_error0,
                  linewidth=1, palette=palette, dashes=False)
ax1.set_ylabel('Error', fontsize=font)
ax1.set_xlabel("Step", fontsize=font)
plt.xlim(0, time_length)
# plt.ylim(-8, 8)
plt.axhline(0, ls='-.', c='k', lw=1, alpha=0.5)
ax1.set_xticks(20 * np.arange(6))
ax1.set_xticklabels(('0', '20', '40', '60', '80', '100'), fontsize=font)
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles=handles, labels=['UKF', 'MHE', r'$\beta$-MHE'], fontsize=15)
plt.yticks(fontsize=font)
plt.xticks(fontsize=font)
plt.savefig("../figures/reversible_reaction/original_1.pdf")

f2 = plt.figure(2)
ax2 = f2.add_axes(axes_pos)
g2 = sns.lineplot(x='Step', y="Error", hue="Algorithm", style="Algorithm", data=pd_error1,
                  linewidth=1, palette=palette, dashes=False
                  )
ax2.set_ylabel('Error', fontsize=font)
ax2.set_xlabel("Step", fontsize=font)
plt.xlim(0, time_length)
plt.yticks(ticks=[-10, -5, 0, 5, 10])
# plt.ylim(-8, 8)
plt.axhline(0, ls='-.', c='k', lw=1, alpha=0.5)
ax2.set_xticks(20 * np.arange(6))
ax2.set_xticklabels(('0', '20', '40', '60', '80', '100'), fontsize=font)
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles=handles, labels=['UKF', 'MHE', r'$\beta$-MHE'], fontsize=15)
plt.yticks(fontsize=font)
plt.xticks(fontsize=font)
plt.savefig("../figures/reversible_reaction/original_2.pdf")
