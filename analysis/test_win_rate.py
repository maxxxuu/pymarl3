import json
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

plot_path = "learning_rate.png"
window = 5

# Currently allow one exp of each map / agent at a time
data_paths = [
    '/Users/zhuofan.xu/Downloads/pymarl_results/8m_vs_9m/spe_qmix/2',
    '/Users/zhuofan.xu/Downloads/pymarl_results/8m_vs_9m/sattpe_qmix/1',
    "/Users/zhuofan.xu/Downloads/pymarl_results/8m_vs_9m/hpn_qmix/6",
    '/Users/zhuofan.xu/Downloads/pymarl_results/8m_vs_9m/attrpe_qmix/4',
    'results/sacred/8m_vs_9m/qmix/1',

]

all_data = []

for data_path in data_paths:

    with open(f'{data_path}/config.json', 'r') as file:
        data = json.load(file)
        config_data = {"map": data["env_args"]["map_name"], "agent": data["name"], }
    with open(f'{data_path}/info.json', 'r') as file:
        data = json.load(file)
        for win_rate, t in zip(data["test_battle_won_mean"], data["test_battle_won_mean_T"]):
            win_rate_data = {"T": t, "test_win_rate": win_rate}

            all_data.append(config_data | win_rate_data)

df = pd.DataFrame(all_data)
df["wind_win_rate"] = df["test_win_rate"].rolling(window, closed="both", center=True).mean()


# test_won_rate = data["test_battle_won_mean"]

plt.figure(figsize=(12, 3))

win_rate_plot = sns.relplot(data=df, x="T", y="test_win_rate", hue="agent", palette="tab10", col="map", kind="line", aspect=3)
# win_rate_plot = sns.relplot(data=df, x="T", y="wind_win_rate", hue="agent", palette="tab10", col="map", kind="line", aspect=3)

plt.savefig(plot_path, dpi=100, bbox_inches="tight")

plt.clf()