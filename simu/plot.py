#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.figure(figsize=[12, 4])
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({
    'lines.linewidth': 1,
    'font.size': 13,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts} \usepackage{amsmath} '
})


use_log = True

Delta = 0.7

fig0_name_null = f"ep-simulation/results_data_simu/K1000N500c5key23333T700Delta{Delta}-alpha0.05-max-null"
fig2_name = f"ep-simulation/results_data_simu/K1000N500c5key23333T700Delta{Delta}-alpha0.05-max-result"
fig0_name_null_seq = f"ep-simulation/results_data_simu/K1000N500c5key23333T700Delta{Delta}-alpha0.05-max-null-seq"


def labelize(name):
    if name == "ars":
        return r"$h_{\mathrm{ars}}$"
    if name == "opt-02":
        return r"$h_{\mathrm{gum},0.2}^{\star}$"
    if name == "opt-005":
        return r"$h_{\mathrm{gum},0.05}^{\star}$"
    if name == "opt-0005":
        return r"$h_{\mathrm{gum},0.005}^{\star}$"
    if name == "opt-0001":
        return r"$h_{\mathrm{gum},0.001}^{\star}$"
    if name == "opt-01":
        return r"$h_{\mathrm{gum},0.1}^{\star}$"
    if name == "opt-001":
        return r"$h_{\mathrm{gum},0.01}^{\star}$"
    elif name == "log":
        return r"$h_{\mathrm{log}}$"
    elif name == "ind-1/e":
        return r"$h_{\mathrm{ind},1/e}$"
    elif name == "ep_01a":
        return r"$E={f}_{\mathrm{mix}}(Y)$"
    elif name == "ep_02a":
        return r"$E=2Y$"
    elif name == "ep_03a":
        return r"$g(p)=-\log(p)$"
    elif name == "ep_04a":
        return r"$E=(1-Y)^{-0.5}-1$"
    elif name == "ep_05a":
        return r"$E=3Y^2$"
    elif name == "ep_unknown":
        return r"$ep_{\mathrm{unknown}}$"
    elif name == "ep_GD0":
        return r"$GD0$"
    elif name == "ep_GD_bw50":
        return r"$GD_{bw100}$"
    elif name == "ep_GD_bw100":
        return r"$GD_{bw200}$"
    elif name == "ep_GDlog":
        return r"$\mathrm{ave}_{\log+GD}$"
    elif name == "ep_GD1":
        return r"$OG$"
    elif name == "ep_GDY":
        return r"$\mathrm{ave}_{2Y+GD}$"
    elif name == "ep_GDY**2":
        return r"$\mathrm{ave}_{3Y^2+GD}$"
    elif name == "ep_GDlogY":
        return r"$\mathrm{ave}_{2Y+\log+GD}$"
    elif name == "ep_GD1log":
        return r"$\mathrm{ave}_{\log+OG}$"
    else:
        raise KeyError(f"{name}")

    

K = 1000
c = 5
# Delta = 0.5
Final_T = 700#40
key = 23333
alpha = 0.05
N_trial = 500
x = np.arange(1,41)
x_l = np.arange(1,Final_T+1)
x_ll = np.arange(1,201)


## For Gumbel-max watermarks

linestyles = ["-", ":", "--", "-."]
colors = ["tab:blue", "tab:orange", "tab:gray", "tab:red", "black", "tab:brown", "tab:purple", "tab:pink", "tab:green", "tab:olive", "tab:cyan"]

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,5))




first = Final_T#700
name = fig2_name
save_dict = json.load(open(name+".json", "r"))

for j, algo in enumerate(["ars", "log", "opt-01", "opt-001",
                          "ep_03a",
                          "ep_GD1","ep_GD1log"]):
    mean = 1-np.array(save_dict[algo])

    if j < 4:

        linestyle = linestyles[(j % 3) + 1]  # 索引1,2,3对应":", "--", "-."
    else:

        linestyle = "-"
    

    ax[0].plot(x_l[3:first], mean[3:first], label=labelize(algo), 
              linestyle=linestyle, color=colors[j % len(colors)])

ax[0].set_title(rf"$H_1, \Delta \sim \mathrm{{U}}  (0.001, {Delta})$")
ax[0].set_ylabel(r"Type II error")
ax[0].set_xlabel(r"Watermarked text length")
if use_log:
    ax[0].set_yscale('log')




first = Final_T#700
name = fig0_name_null
save_dict = json.load(open(name+".json", "r"))

for j, algo in enumerate(["ars", "log", "opt-01", "opt-001",
                          "ep_03a",
                          "ep_GD1","ep_GD1log"]):
    mean = np.array(save_dict[algo])

    if j < 4:

        linestyle = linestyles[(j % 3) + 1]  # 索引1,2,3对应":", "--", "-."
    else:

        linestyle = "-"
    

    ax[1].plot(x_l[3:first], mean[3:first], label=labelize(algo), 
              linestyle=linestyle, color=colors[j % len(colors)])

ax[1].set_title(r"$H_0$")
ax[1].axhline(y=0.05, color="black", linestyle="dotted")
ax[1].set_ylabel(r"Type I error")
ax[1].set_xlabel(r"Unwatermarked text length")




first = Final_T#700
name = fig0_name_null_seq
save_dict = json.load(open(name+".json", "r"))

for j, algo in enumerate(["ars", "log", "opt-01", "opt-001",
                          "ep_03a",
                          "ep_GD1","ep_GD1log"]):
    mean = np.array(save_dict[algo])

    if j < 4:

        linestyle = linestyles[(j % 3) + 1]  # 索引1,2,3对应":", "--", "-."
    else:

        linestyle = "-"
    

    ax[2].plot(x_l[3:first], mean[3:first], label=labelize(algo), 
              linestyle=linestyle, color=colors[j % len(colors)])

ax[2].set_title(r"$H_0$")
ax[2].axhline(y=0.05, color="black", linestyle="dotted")
ax[2].set_ylabel(r"Sequential Type I error")
ax[2].set_xlabel(r"Unwatermarked text length")
# ax[2].set_ylim(ymin=0.001)
# if use_log:
#     ax[2].set_yscale('log')


# ax[2].legend()


handles, labels = ax[0].get_legend_handles_labels()


ax[1].legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15),
             ncol=2, fancybox=True, shadow=True, fontsize=12,
             frameon=True, handlelength=2.5, handletextpad=0.5)


plt.tight_layout()
plt.subplots_adjust(top=0.85)  

# plt.tight_layout()
plt.savefig(f'ep-simulation/results_data_simu/simu-GD-temp{Delta}.pdf', dpi=300)
