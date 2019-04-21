import pandas as pd
import matplotlib.pyplot as plt
plot_fig = 4

if plot_fig == 1:  # Q Learner
    df = pd.read_csv('q-learning.csv', index_col=0)
    df_qLearn = df.iloc[:,:1]
    df_qLearn.plot(x=df_qLearn.index, y=df_qLearn.columns, title='Q-learner', legend=None, color='k', linewidth=0.2)
    plt.xlabel('Simulation Iteration', fontsize=12)

    plt.ylabel('Q-value Difference', fontsize=12)

    plt.ylim((0., 0.5))
    plt.show()
elif plot_fig == 2: # Friend-Q
    df = pd.read_csv('friend-q.csv', index_col=0)
    df_friendQ = df.iloc[:,:1]
    df_friendQ.plot(x=df_friendQ.index, y=df_friendQ.columns, title='Friend-Q', legend=None, color='k', linewidth=0.4)
    plt.xlabel('Simulation Iteration', fontsize=12)

    plt.ylabel('Q-value Difference', fontsize=12)

    plt.ylim((0., 0.5))
    plt.show()
elif plot_fig == 3:  # Foe-Q
    df = pd.read_csv('foe-q.csv', index_col=0)
    df_foeQ = df.iloc[:,:1]
    df_foeQ.plot(x=df_foeQ.index, y=df_foeQ.columns, title='Foe-Q', legend=None, color='k', linewidth=0.4)
    plt.xlabel('Simulation Iteration', fontsize=12)

    plt.ylabel('Q-value Difference', fontsize=12)

    plt.ylim((0., 0.5))
    plt.show()
elif plot_fig == 4: # Correlated-Q
    df = pd.read_csv('ce-q.csv', index_col=0)
    df_ceQ = df.iloc[:,:1]
    df_ceQ.plot(x=df_ceQ.index, y=df_ceQ.columns, title='Correlated-Q', legend=None, color='k', linewidth=0.4)
    plt.xlabel('Simulation Iteration', fontsize=12)

    plt.ylabel('Q-value Difference', fontsize=12)

    plt.ylim((0., 0.5))
    plt.show()
