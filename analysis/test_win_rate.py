import json
import os
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

plot_path = "plots"

if not os.path.exists(plot_path):
    os.makedirs(plot_path)

# window = 5
test_interval = 10000

# Currently allow one exp of each map / agent at a time
data_paths = [
    # 'results/sacred/8m_vs_9m/qmix/1',
    # '/Users/zhuofan.xu/Downloads/pymarl_results/8m_vs_9m/qmix/1',
    '/Users/zhuofan.xu/Downloads/pymarl_results/8m_vs_9m/spe_qmix/2',
    '/Users/zhuofan.xu/Downloads/pymarl_results/8m_vs_9m/sattpe_qmix/1',
    '/Users/zhuofan.xu/Downloads/pymarl_results/8m_vs_9m/sattpe_qmix/3',
    '/Users/zhuofan.xu/Downloads/pymarl_results/8m_vs_9m/hpn_qmix/4',
    "/Users/zhuofan.xu/Downloads/pymarl_results/8m_vs_9m/hpn_qmix/6",
    # '/Users/zhuofan.xu/Downloads/pymarl_results/8m_vs_9m/attrpe_qmix/4',
    '/Users/zhuofan.xu/Downloads/pymarl_results/8m_vs_9m/sattpe1_qmix/3',
    # '/Users/zhuofan.xu/Downloads/pymarl_results/8m_vs_9m/spe_light_qmix/2',
    # '/Users/zhuofan.xu/Downloads/pymarl_results/8m_vs_9m/spe_peqmix/1',
    # '/Users/zhuofan.xu/Downloads/pymarl_results/8m_vs_9m/spe_medium_qmix/1',

    '/Users/zhuofan.xu/Downloads/pymarl_results/2c_vs_64zg/hpn_qmix/1',
    # '/Users/zhuofan.xu/Downloads/pymarl_results/2c_vs_64zg/spe_qmix/3',

    '/Users/zhuofan.xu/Downloads/pymarl_results/6h_vs_8z/spe_qmix/1',
    # '/Users/zhuofan.xu/Downloads/pymarl_results/6h_vs_8z/spe_qmix/2',
    # '/Users/zhuofan.xu/Downloads/pymarl_results/6h_vs_8z/spe_qmix/4',
    # '/Users/zhuofan.xu/Downloads/pymarl_results/6h_vs_8z/spe_qmix/8',
    '/Users/zhuofan.xu/Downloads/pymarl_results/6h_vs_8z/spe_qmix/9',
    '/Users/zhuofan.xu/Downloads/pymarl_results/6h_vs_8z/spe_qmix/10',
    '/Users/zhuofan.xu/Downloads/pymarl_results/6h_vs_8z/hpn_qmix/1',
    '/Users/zhuofan.xu/Downloads/pymarl_results/6h_vs_8z/hpn_qmix/2',
    # '/Users/zhuofan.xu/Downloads/pymarl_results/6h_vs_8z/hpn_qmix/4',
    # '/Users/zhuofan.xu/Downloads/pymarl_results/6h_vs_8z/hpn_qmix/5',
    '/Users/zhuofan.xu/Downloads/pymarl_results/6h_vs_8z/hpn_qmix/6',
    # '/Users/zhuofan.xu/Downloads/pymarl_results/6h_vs_8z/hpn_qmix/7',
    '/Users/zhuofan.xu/Downloads/pymarl_results/6h_vs_8z/deepset_qmix/1',
    '/Users/zhuofan.xu/Downloads/pymarl_results/6h_vs_8z/deepset_qmix/2',
    '/Users/zhuofan.xu/Downloads/pymarl_results/6h_vs_8z/deepset_qmix/3',
    '/Users/zhuofan.xu/Downloads/pymarl_results/6h_vs_8z/spe_light_qmix/1',
    '/Users/zhuofan.xu/Downloads/pymarl_results/6h_vs_8z/qmix/1',

    # '/Users/zhuofan.xu/Downloads/pymarl_results/3s5z_vs_3s6z/hpn_qmix/1',
    '/Users/zhuofan.xu/Downloads/pymarl_results/3s5z_vs_3s6z/hpn_qmix/3',
    '/Users/zhuofan.xu/Downloads/pymarl_results/3s5z_vs_3s6z/hpn_qmix/4',
    # '/Users/zhuofan.xu/Downloads/pymarl_results/3s5z_vs_3s6z/hpn_qmix/5',
    '/Users/zhuofan.xu/Downloads/pymarl_results/3s5z_vs_3s6z/hpn_qmix/6',
    '/Users/zhuofan.xu/Downloads/pymarl_results/3s5z_vs_3s6z/spe_qmix/2',
    '/Users/zhuofan.xu/Downloads/pymarl_results/3s5z_vs_3s6z/spe_qmix/3',
    # '/Users/zhuofan.xu/Downloads/pymarl_results/3s5z_vs_3s6z/spe_qmix/4',
    # '/Users/zhuofan.xu/Downloads/pymarl_results/3s5z_vs_3s6z/spe_qmix/5',
    '/Users/zhuofan.xu/Downloads/pymarl_results/3s5z_vs_3s6z/spe_qmix/6',
    # '/Users/zhuofan.xu/Downloads/pymarl_results/3s5z_vs_3s6z/spe_qmix/8',
    '/Users/zhuofan.xu/Downloads/pymarl_results/3s5z_vs_3s6z/sattpe_qmix/1',
    '/Users/zhuofan.xu/Downloads/pymarl_results/3s5z_vs_3s6z/deepset_qmix/1',
    '/Users/zhuofan.xu/Downloads/pymarl_results/3s5z_vs_3s6z/deepset_qmix/2',
    '/Users/zhuofan.xu/Downloads/pymarl_results/3s5z_vs_3s6z/deepset_qmix/3',

    '/Users/zhuofan.xu/Downloads/pymarl_results/MMM2/hpn_qmix/1',
    '/Users/zhuofan.xu/Downloads/pymarl_results/MMM2/hpn_qmix/3',
    '/Users/zhuofan.xu/Downloads/pymarl_results/MMM2/hpn_qmix/4',
    # '/Users/zhuofan.xu/Downloads/pymarl_results/MMM2/spe_qmix/4',
    '/Users/zhuofan.xu/Downloads/pymarl_results/MMM2/spe_qmix/5',
    '/Users/zhuofan.xu/Downloads/pymarl_results/MMM2/spe_qmix/6',
    '/Users/zhuofan.xu/Downloads/pymarl_results/MMM2/spe_qmix/7',
    '/Users/zhuofan.xu/Downloads/pymarl_results/MMM2/deepset_qmix/1',
    '/Users/zhuofan.xu/Downloads/pymarl_results/MMM2/deepset_qmix/2',
    '/Users/zhuofan.xu/Downloads/pymarl_results/MMM2/deepset_qmix/3',


]

models = ['spe_qmix', 'deepset_qmix', 'hpn_qmix', 'sattpe1_qmix', 'sattpe_qmix']
maps = ['8m_vs_9m', '6h_vs_8z', 'MMM2', '3s5z_vs_3s6z']
dict_rename = {
    'spe_qmix': 'EFPENN',
    'deepset_qmix': 'DeepSet',
    'hpn_qmix': 'HPN',
    'sattpe1_qmix': 'sattpe1_qmix',
    'sattpe_qmix': 'sattpe_qmix'}
order_label = ['EFPENN', 'DeepSet', 'HPN', 'sattpe_qmix', 'sattpe1_qmix']

all_data = []

for data_path in data_paths:

    with open(f'{data_path}/config.json', 'r') as file:
        data = json.load(file)
        config_data = {"map": data["env_args"]["map_name"], "agent": data["name"], }
    with open(f'{data_path}/info.json', 'r') as file:
        print(data_path)
        data = json.load(file)
        for win_rate, t in zip(data["test_battle_won_mean"], data["test_battle_won_mean_T"]):
            win_rate_data = {"T": (t // test_interval) * test_interval, "test_win_rate": win_rate}

            all_data.append(config_data | win_rate_data)

df = pd.DataFrame(all_data)
df = df[df["agent"].isin(models) & df["map"].isin(maps)].replace(dict_rename)
# df["wind_win_rate"] = df["test_win_rate"].rolling(window, closed="both", center=True).mean()


# test_won_rate = data["test_battle_won_mean"]
sns.set_style("whitegrid",
              # {"grid.color": ".4"}
              )
plt.rcParams.update({'font.size': 18})

for map in df["map"].unique():

    # plt.figure(figsize=(9, 3))

    win_rate_plot = sns.relplot(
        data=df[df["map"] == map], x="T", y="test_win_rate", hue="agent", hue_order=order_label,
        palette="hls", col="map", kind="line",
        aspect=2.5,
        estimator="median", errorbar="pi")
    # win_rate_plot = sns.relplot(data=df, x="T", y="wind_win_rate", hue="agent", palette="tab10", col="map", kind="line", aspect=3)

    plt.savefig(f"{plot_path}/{map}.pdf", bbox_inches="tight", facecolor=(1,1,1,0), dpi=500,)

    plt.clf()


# plt.figure(figsize=(12, 3))

win_rate_plot = sns.relplot(
    data=df, x="T", y="test_win_rate", hue="agent", hue_order=order_label,
    palette="hls", col=None, row="map", row_order=['3s5z_vs_3s6z', '6h_vs_8z', 'MMM2'], kind="line", aspect=4, estimator="median", errorbar="pi", )

# win_rate_plot = sns.relplot(
#     data=df, x="T", y="test_win_rate", hue="agent", hue_order=order_label,
#     palette="hls", col="map", col_order=['3s5z_vs_3s6z', '6h_vs_8z', 'MMM2'], kind="line", aspect=4, estimator="median", errorbar="pi", )


# Adjust legend to be a single row at the top
handles, labels = win_rate_plot._legend.legend_handles, [t.get_text() for t in win_rate_plot._legend.texts]  # Get legend items

# Remove the default legend
win_rate_plot._legend.remove()

# Create a new legend with all items in one row
win_rate_plot.fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=len(labels), frameon=False)



plt.savefig(f"{plot_path}/smac_result.png", bbox_inches="tight", facecolor=(1,1,1,0), dpi=300,)

plt.clf()