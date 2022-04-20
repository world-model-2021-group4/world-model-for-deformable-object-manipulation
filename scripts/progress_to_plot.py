import pandas as pd
import matplotlib.pyplot as plt
import argparse

def progress_to_plat(csv_path, fig_path, title):
    df = pd.read_csv(csv_path)
    # fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 7))
    # df[['num_episodes', 'train_rewards']].plot(x='num_episodes', ax=axes[0, 0])
    # df[['num_episodes', 'observation_loss']].plot(x='num_episodes', ax=axes[0, 1], logy=True)
    # df[['num_episodes', 'kl_loss']].plot(x='num_episodes', ax=axes[1, 0])
    # df[['num_episodes', 'reward_loss']].plot(x='num_episodes', ax=axes[1, 1], logy=True)
    df[['num_episodes', 'train_rewards', 'observation_loss', 'kl_loss', 'reward_loss']].plot.line(title=title, x='num_episodes', subplots=True, layout=(2, 2), figsize=(9, 6))
    # plt.show()
    plt.savefig(fig_path)
    plt.close('all')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', default='/data/progress.csv', type=str)
    parser.add_argument('-f', '--fig', default='/data/plot.png', type=str)
    parser.add_argument('-t', '--title', default='FoldCloth', type=str)
    args = parser.parse_args()
    csv_path = args.data
    fig_path = args.fig
    title = args.title
    progress_to_plat(csv_path, fig_path, title)
