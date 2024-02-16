import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_accuracie_simple( training_log, testing_log, log_file_path):
    # Load the log.csv file
    log_data = pd.read_csv(log_file_path)

    # Extract the required columns
    reward_mean = log_data['reward_mean']
    env = log_data['env']

    no_of_values = 4
    norm = mpl.colors.Normalize(vmin=min([0, no_of_values]), vmax=max([0, no_of_values]))
    cmap_obj = mpl.cm.get_cmap('Set1')  # tab20b tab20
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj)

    log = testing_log
    training_log.switch_trialxxbatch.append(training_log.stamps[-1])
    num_tasks = len(config.tasks)
    already_seen = []
    title_label = 'Training tasks sequentially ---> \n    ' + config.exp_name
    max_x = training_log.stamps[-1]
    fig, axes = plt.subplots(num_tasks + 4, 1, figsize=[9, 7])

    # Plotting code
    ax = axes[0]
    ax.plot(log_data.index, reward_mean, color='blue')
    ax.set_ylabel('Reward Mean')

    for i in range(len(env)):
        ax.axvspan(i, i + 1, facecolor=env[i], alpha=0.2)

    # Rest of the plotting code goes here

    final_accuracy_average = np.mean(list(testing_log.accuracies[-1].values()))
    identifiers = 9  # f'{training_log.stamps[-1]}_{final_accuracy_average:1.2f}'
    plt.savefig('./files/' + config.exp_name + f'/acc_summary_{config.exp_signature}_{identifiers}.jpg', dpi=300)


def plot_accuracies( config, training_log, testing_log):
   
    no_of_values = len(config.tasks)
    norm = mpl.colors.Normalize(vmin=min([0,no_of_values]), vmax=max([0,no_of_values]))
    cmap_obj = mpl.cm.get_cmap('Set1') # tab20b tab20
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj)

    log = testing_log
    training_log.switch_trialxxbatch.append(training_log.stamps[-1])
    num_tasks = len(config.tasks)
    already_seen =[]
    title_label = 'Training tasks sequentially ---> \n    ' + config.exp_name
    max_x = training_log.stamps[-1]
    fig, axes = plt.subplots(num_tasks+4,1, figsize=[9,7])
    for logi in range(num_tasks):
            ax = axes[ logi ] # log i goes to the col direction -->
            ax.set_ylim([-0.1,1.1])
    #         ax.axis('off')
            ax.plot(testing_log.stamps, [test_acc[logi] for test_acc in testing_log.accuracies], linewidth=1)
            ax.plot(testing_log.stamps, np.ones_like(testing_log.stamps)*0.5, ':', color='grey', linewidth=1)
            ax.set_ylabel(config.human_task_names[logi].replace('class', 'Task '), fontdict={'color': cmap.to_rgba(logi)})
            ax.set_xlim([0, max_x])
            if (logi == num_tasks-1) and config.use_cognitive_observer and config.train_cog_obs_on_recent_trials: # the last subplot, put the preds from cog_obx
                cop = np.stack(training_log.cog_obs_preds).reshape([-1,100,15])
                cop_colors = np.argmax(cop, axis=-1).mean(-1)
                for ri in range(max_x-2):
                    if ri< len(cop_colors):ax.axvspan(ri, ri+1, color =cmap.to_rgba(cop_colors[ri]) , alpha=0.2)
            else:            
                for ri in range(len(training_log.switch_trialxxbatch)-1):
                    ax.axvspan(training_log.switch_trialxxbatch[ri], training_log.switch_trialxxbatch[ri+1], color =cmap.to_rgba(training_log.switch_task_id[ri]) , alpha=0.2)
    for ti, id in enumerate(training_log.switch_task_id):
        if id not in already_seen:
            already_seen.append(id)
            task_name = config.human_task_names[id]
            axes[0].text(training_log.switch_trialxxbatch[ti], 1.3, task_name.replace('class', 'Task '), color= cmap.to_rgba(id) )

    lens = [len(tg) for tg in training_log.gradients]
    m = min(lens)
    training_log.gradients = [tg[:m] for tg in training_log.gradients]
    try:
        gs = np.stack(training_log.gradients)
    except:
        pass

    glabels =  ['inp_w', 'inp_b', 'rnn_w', 'rnn_b', 'out_w', 'out_b']
    ax = axes[num_tasks+0]
    gi =0
    ax.plot(training_log.stamps, gs[:,gi+1], label= glabels[gi])
    gi =2
    ax.plot(training_log.stamps, gs[:,gi+1], label= glabels[gi])
    gi = 4
    ax.plot(training_log.stamps, gs[:,gi+1], label= glabels[gi])
    ax.legend()
    ax.set_xlim([0, max_x])
    ax = axes[num_tasks+1]
    ax.plot(testing_log.stamps,  [np.mean(list(la.values())) for la in testing_log.accuracies] )
    ax.set_ylabel('avg acc')

    ax.plot(testing_log.stamps, np.ones_like(testing_log.stamps)*0.9, ':', color='grey', linewidth=1)
    ax.set_xlim([0, max_x])
    ax.set_ylim([-0.1,1.1])

    ax = axes[num_tasks+2]
    ax.plot(training_log.switch_trialxxbatch[:-1], training_log.trials_to_crit)
    ax.set_ylabel('ttc')
    ax.set_xlim([0, max_x])

    ax = axes[num_tasks+3]
    if len(training_log.stamps) == len(training_log.frustrations):
        ax.plot(training_log.stamps, training_log.frustrations)
    ax.set_ylabel('frust')
    ax.set_xlim([0, max_x])


    final_accuracy_average = np.mean(list(testing_log.accuracies[-1].values()))
    identifiers = 9 # f'{training_log.stamps[-1]}_{final_accuracy_average:1.2f}'
    plt.savefig('./files/'+ config.exp_name+f'/acc_summary_{config.exp_signature}_{identifiers}.jpg', dpi=300)



if __name__ == '__main__':
    import pandas as pd
    import tensorboardX
    import pandas as pd
    import torch
    import matplotlib.pyplot as plt
    # Load the status.pt file
    #close all the figures
    plt.close('all')
    file_list = ['20240214ppo2swap1281','20240214pposwap1281']
    data_list = []
    frame_list = []
    scalar_list = []
    fig, ax = plt.subplots(figsize=[9,7])

    for name in file_list:
        data = pd.read_csv('/Users/bytedance/Documents/PHD/rl-starter-files/storage/'+name+'/log.csv')

    # Extract the desired data from the status object
    # Extract the return_mean and frames columns

        data = data[data['rreturn_mean'] != 'rreturn_mean']
        return_mean = pd.to_numeric(data['rreturn_mean'], errors='coerce')
        frames = pd.to_numeric(data['frames'], errors='coerce')
        # fig, axes = plt.subplots(1,1, figsize=[9,7])
        frame_list.append(frames)
        data_list.append(data)

        line, = ax.plot(frames, return_mean,label=name)
        # line_color = line.get_color()
        # # store the color of the line pair with name to use for later legend
        # scalar_list.append((name,line_color))

        env_values = data['env']
        env_label = data['env_label']
        frames = pd.to_numeric(data['frames'], errors='coerce')
        unique_envs = env_label.unique()
        print('unique_envs',unique_envs)
        unique_labels = env_values.unique()
        env_names = ["DoorKeyEnv", "CrossingEnv", "DynamicObstaclesEnv", "CrossinggoodlavaEnv"]
        colors = ['red', 'blue', 'green', 'yellow']  # Add more colors if needed
        # Create a dictionary to map environments to colors
        env_color_map = dict(zip(env_names, colors))
        # Create a dictionary to map numbers to environment names
        number_to_name_map = {"DoorKeyEnv":1,  "CrossingEnv":0,  "DynamicObstaclesEnv":3,  "CrossinggoodlavaEnv":2 }

        ax.axvspan(0, frames[0], facecolor=env_color_map.get('DoorKeyEnv'), alpha=.1)
        for env in unique_envs:
            color = env_color_map.get(env)  
            print('env',env,color)
            # env_indice = number_to_name_map[env]
            #get all frame indices for this env
            env_indices = list(data[data['env_label'] == env].index)
            print('env_indices',env_indices)
            # print item's whole rwo on indice 44 in frams
            for i in range(len(env_indices)):
                # print('current env_indice:',i,env_indices[i])
                # if this indice is the last one inframe, do not plot
                print('-1',frames[:-1])
                if env_indices[i] == frames.index[-1]:
                    xmin = frames[env_indices[i]]
                    xmax = frames[env_indices[i]]
                else:
                    xmin = frames[env_indices[i]]
                    xmax = frames[env_indices[i]+1]
                print('current env_indice:',i,env_indices[i],'the color section is',xmin,xmax,'of',color)
                ax.axvspan(xmin,xmax, facecolor=color, alpha=.2)
                # plt.text(frames[env_indices[i]], 1, env_indices[i], fontsize=8)

    first_legend = ax.legend(loc='lower left')
    ax.add_artist(first_legend)
    # Create a list of patches for the legend
    patches = []
    for env in unique_envs:
        color = env_color_map.get(env)
        patch = mpl.patches.Patch(color=color, label=env)
        patches.append(patch)
    # for name in file_list:
    #     patch = mpl.patches.Patch( label=name)
    #     patches.append(patch)

    # Add the legend to the plot
    ax.legend(handles=patches)


    plt.title('Mean Return Over Time'+name)
    plt.xlabel('Frames')
    plt.ylabel('Mean Return')
    plt.grid(True)
    print('showing',unique_envs)
    plt.show()
