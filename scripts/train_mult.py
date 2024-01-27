import argparse
import os
import time
import datetime
import torch_ac
import tensorboardX
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import sys

import utils
from utils import device
from model import ACModel  as BaseACModel
# from model_modified import ACModel
from model_modified2 import ACModel

from ali import AgentNetwork
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
import pandas as pd
from array2gif import write_gif

import os
import os
# Parse arguments

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument("--algo", required=True,
                    help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=10,
                    help="number of updates between two saves (default: 10, 0 means no saving)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10**7,
                    help="number of frames of training (default: 1e7)")

# Parameters for main algorithm
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size for PPO (default: 256)")
parser.add_argument("--frames-per-proc", type=int, default=None,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate (default: 0.001)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-8,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model to handle text input")
parser.add_argument("--context_size", type=int, default=4,
                    help="add a context latent z into the model")
parser.add_argument("--num-models", type=int, default=5, 
                    help="number of models to train with different seeds (default: 1)")
parser.add_argument("--interval", type=int, default=999,
                    help="number of frames between swapping environments (default: 1000000)")

parser.add_argument("--gif", type=str, default=None,
                    help="store output as gif with the given filename")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability (default: False)")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")

if __name__ == "__main__":
    args = parser.parse_args()



    args.mem = args.recurrence > 1
    rng = np.random.RandomState(args.seed)
    swap_seq = rng.randint(0, 100, size=int(args.frames/args.interval))
    env_names = ["doorkey", "cross", "obstacle", "goodlava"]

    # make swap a fixed random sequence like [0,1,2,3,0,1,2,3,0,1,2,3] with length of args.frames/args.interval, every run should have the same swap sequence
    # swap_seq = [0,1,2,3,0,1,2,3,0,1,2,3]
    # swap_seq = [37, 86, 30, 50, 20, 83, 60, 52, 75, 75, 27, 28, 91, 38, 15, 36, 43, 50, 97, 80, 14, 36, 62, 89, 40, 78, 89, 23, 50, 54, 3, 85, 0, 56, 5, 35, 16, 28, 21, 81, 94, 69, 86, 21, 94, 21, 78, 64, 60, 43, 65, 43, 25, 18, 37, 20, 81, 89, 89, 37, 13, 58, 44, 4, 83, 29, 19, 65, 18, 74, 96, 55, 60, 10, 54, 62, 59, 6, 59, 99, 60, 85, 81, 43, 58, 94, 60, 8, 70, 64, 42, 2, 98, 53, 2, 62, 11, 40, 58, 40, 4, 32, 26, 40, 78, 43, 17, 42, 71, 28, 77, 48, 98, 20, 93, 85, 95, 0, 9, 11, 21, 17, 20, 91, 51, 15, 96, 76, 3, 82, 62, 47, 25, 73, 36, 7, 96, 96, 51, 0, 9, 63, 97, 28, 56, 80, 51, 56, 58, 4, 91, 66, 56, 53, 30, 99, 79, 60, 57, 6, 95, 79, 54, 42, 2, 37, 67, 91, 47, 15, 29, 64, 12, 11, 77, 73, 51, 40, 79, 82, 42, 37, 71, 85, 0, 29, 65, 12, 78, 73, 31, 77, 12, 86, 96, 44, 49, 97, 25, 4]

    for model_idx in range(args.num_models):
        # Adjust seed for each model
        current_seed = args.seed + model_idx
        utils.seed(current_seed)
        date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        default_model_name = f"{args.env}_{args.algo}_seed{current_seed}_{date}"
        model_name = args.model+str(current_seed) or default_model_name
        model_dir = utils.get_model_dir(model_name)
    

        # Load loggers and Tensorboard writer

        txt_logger = utils.get_txt_logger(model_dir)
        csv_file, csv_logger = utils.get_csv_logger(model_dir)
        tb_writer = tensorboardX.SummaryWriter(model_dir)

        # Log command and all script arguments

        txt_logger.info("{}\n".format(" ".join(sys.argv)))
        txt_logger.info("{}\n".format(args))

        # Set seed for all randomness sources

        # utils.seed(args.seed)

        # Set device

        txt_logger.info(f"Device: {device}\n")

        # Load environments

        envs = []
        #generate a random swap sequence of length frame /args.interval
        print(swap_seq)
        if args.env == 'MiniGrid-BlendCrossDoorkey-v0':
            for i in range(args.procs):
                envs.append(utils.make_env(args.env, args.seed + 10000 * i,t=args.interval,swap_seq=swap_seq))
        else:
            for i in range(args.procs):
                envs.append(utils.make_env(args.env, args.seed + 10000 * i))
        txt_logger.info("Environments loaded\n")


        # Load training status

        try:
            status = utils.get_status(model_dir)
        except OSError:
            status = {"num_frames": 0, "update": 0}
        txt_logger.info("Training status loaded\n")

        # Load observations preprocessor

        obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
        if "vocab" in status:
            preprocess_obss.vocab.load_vocab(status["vocab"])
        txt_logger.info("Observations preprocessor loaded")

        # Load model
        args.context_size = 4
        if args.algo == "ppo2":

            acmodel = ACModel(obs_space, envs[0].action_space, args.mem, args.text,args.context_size)
            # acmodel = AgentNetwork(obs_space, envs[0].action_space, args.mem, args.text)
            if "model_state" in status:
                acmodel.load_state_dict(status["model_state"])
            # Initialize world model's weights if they were not loaded
            if 'world_model' in vars(acmodel).keys():  # Check if world_model attribute exists
            # Define the initialization function
                def init_weights(m):
                    if type(m) ==torch.nn.Linear:
                        torch.nn.init.xavier_uniform_(m.weight)
                        m.bias.data.fill_(0.01)

            # Apply the initialization to the world model
                acmodel.world_model.apply(init_weights)
            txt_logger.info("World model weights initialized")
            acmodel.to(device)
            txt_logger.info("Model loaded\n")
            txt_logger.info("{}\n".format(acmodel))
            try:
                txt_logger.info("{}\n".format( envs[0].unwrapped.envs))
            except:
                txt_logger.info("{}\n".format(envs[0]))
            try:
                txt_logger.info("{}\n".format( envs[0].unwrapped.current_env))
            except:
                txt_logger.info("{}\n".format(envs[0]))
        else:
            acmodel = BaseACModel(obs_space, envs[0].action_space, args.mem, args.text)
            # acmodel = AgentNetwork(obs_space, envs[0].action_space, args.mem, args.text)
            if "model_state" in status:
                acmodel.load_state_dict(status["model_state"])
            # Initialize world model's weights if they were not loaded
            if 'world_model' in vars(acmodel).keys():  # Check if world_model attribute exists
            # Define the initialization function
                def init_weights(m):
                    if type(m) ==torch.nn.Linear:
                        torch.nn.init.xavier_uniform_(m.weight)
                        m.bias.data.fill_(0.01)

            # Apply the initialization to the world model
            acmodel.to(device)
            txt_logger.info("Model loaded\n")
            txt_logger.info("{}\n".format(acmodel))
            try:
                txt_logger.info("{}\n".format( envs[0].unwrapped.envs))
            except:
                txt_logger.info("{}\n".format(envs[0]))
            try:
                txt_logger.info("{}\n".format( envs[0].unwrapped.current_env))
            except:
                txt_logger.info("{}\n".format(envs[0]))



        # Load algo

        if args.algo == "a2c":
            algo = torch_ac.A2CAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                    args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                    args.optim_alpha, args.optim_eps, preprocess_obss)
        elif args.algo == "ppo":
            algo = torch_ac.PPOAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                    args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                    args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)
        elif args.algo == "ppo2":
            algo = torch_ac.PPO2Algo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                    args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                    args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)
        else:
            raise ValueError("Incorrect algorithm name: {}".format(args.algo))

        if "optimizer_state" in status:
            algo.optimizer.load_state_dict(status["optimizer_state"])
        txt_logger.info("Optimizer loaded\n")

        # Train model

        num_frames = status["num_frames"]
        update = status["update"]
        start_time = time.time()

        while num_frames < args.frames:
            print("num_frames: ",num_frames)
            # Update model parameters
            update_start_time = time.time()

            # inject current frame into env
            # if args.env == 'MiniGrid-BlendCrossDoorkey-v0':
            #     if num_frames % args.interval == 0:
            #         # swap env and maintain agent positon
            #         #get current observation
            #         obs = envs[0].get_obs()
            #         envs[0].swap_env(obs)
            #         print(envs[0].get_env_name())
            #         env_swap = 1


            if args.env == 'MiniGrid-BlendCrossDoorkey-v0':
                if args.algo == "ppo2":
                    exps, logs1 = algo.collect_experiences_latent()
                else:
                    exps, logs1 = algo.collect_experiences()
            else:
                if args.algo == "ppo2":
                    exps, logs1 = algo.collect_experiences_latent(latent_z=torch.tensor([1, 0.25, 0.25, 0.25]))
                else:
                    exps, logs1 = algo.collect_experiences()

            logs2 = algo.update_parameters(exps)
            logs = {**logs1, **logs2}
            update_end_time = time.time()

            num_frames += logs["num_frames"]
            update += 1

            # Print logs

            if update % args.log_interval == 0:
                fps = logs["num_frames"] / (update_end_time - update_start_time)
                duration = int(time.time() - start_time)
                return_per_episode = utils.synthesize(logs["return_per_episode"])
                rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
                num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

                header = ["update", "frames", "FPS", "duration"]
                data = [update, num_frames, fps, duration]
                header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
                data += rreturn_per_episode.values()
                header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
                data += num_frames_per_episode.values()
                header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm","world_loss","env"]
                if args.algo == "ppo2":
                    data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"],logs["world_loss"],logs["env"]]
                    txt_logger.info(
                    "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}| wl {:.3f} env {:d}"
                    .format(*data))
                    # txt_logger.info(algo.latent_z)

                else:
                    data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"],logs["env"]]
                    txt_logger.info(
                    "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}  env {:d}"
                    .format(*data))



                header += ["return_" + key for key in return_per_episode.keys()]
                data += return_per_episode.values()
                # header+=[ "world_loss","env"]


                if status["num_frames"] == 0:
                    csv_logger.writerow(header)
                csv_logger.writerow(data)
                csv_file.flush()
                
                for field, value in zip(header, data):
                    tb_writer.add_scalar(field, value, num_frames)

            # Save status

            if args.save_interval > 0 and update % args.save_interval == 0:
                status = {"num_frames": num_frames, "update": update,
                        "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
                if hasattr(preprocess_obss, "vocab"):
                    status["vocab"] = preprocess_obss.vocab.vocab
                utils.save_status(status, model_dir)
                txt_logger.info("Status saved")

            # save gif
            if args.save_interval > 0 and update % args.save_interval == 0:
                    frames = []
                    # result = algo.save_gif(update, model_dir)
                    env , env_idx = algo.get_env()
                    if args.algo == "ppo2":
                        agent = utils.Agent2(env.observation_space, env.action_space, model_dir,
                    argmax=args.argmax, use_memory=args.memory, use_text=args.text)
                    else:
                        agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                    argmax=args.argmax, use_memory=args.memory, use_text=args.text)
                    obs, _ = env.reset()
                    env_name =env.get_env_name
                    print("Environment loaded\n")
                    count = 0
                    # obs, _ = env.reset()
                    if args.gif:
                        while True:
                            # env.render()
                            frames.append(np.moveaxis(env.get_frame(), 2, 0))
                            if args.algo == "ppo2":
                                try:
                                    latent_z = env.get_ground_truth_latent_z()
                                except:
                                    latent_z = torch.tensor([1, 0.25, 0.25, 0.25])
                                action = agent.get_action(obs,latent_z=latent_z)
                            else:
                                action = agent.get_action(obs)

                            obs, reward, terminated, truncated, _ = env.step(action)
                            done = terminated | truncated
                            agent.analyze_feedback(reward, done)
                            count += 1

                            if count > 1000:
                                print(count)
                                break
                            if done:
                                print('done')
                                break
                    print("Saving gif... ", end="")

                    env_name = env_names[env_idx]
                    print(env_name)

                    gif_path = os.path.join(model_dir, f"{str(update)}{str(env_name)} {str(env_idx)}:{str(num_frames)}-{str(count)}.gif")
                    write_gif(np.array(frames), gif_path, fps=1/args.pause)
                    print(gif_path)
                    txt_logger.info("gif saved")
        swap_id = 0
        env_colors = {"doorkey": "red", "cross": "blue", "obstacle": "green", "goodlava": "yellow"}  # Example colors for each environment        swap_seq_data=[]

        for i in range(args.frames):
            # swap at every interval *16 frames

            if i % (args.interval * 16) == 0:
                swap_id = swap_id + 1
                
            env_name = env_names[swap_seq[swap_id] % 4]
            # print("num_frames:", i, "env:", env_name)

            env_id = env_names.index(env_name)
            tb_writer.add_scalar('Environment', env_id, i)

# import matplotlib.pyplot as plt
# # Top Plot: Environment Indication
# env_names = ["doorkey", "cross", "obstacle", "goodlava"]
# env_colors = {"doorkey": "red", "cross": "blue", "obstacle": "green", "goodlava": "yellow"}

# env_indicator = []
# for i in range(args.frames):
#     swap_id = i // (args.interval * 16)
#     env_name = env_names[swap_seq[swap_id] % 4]
#     env_indicator.append(env_colors[env_name])

# plt.subplot(2, 1, 1)
# plt.bar(range(args.frames), [1] * args.frames, color=env_indicator)
# plt.xlabel("Frame")
# plt.ylabel("Environment")
# plt.title("Environment Indication")

# # Bottom Plot: Rreturn
# df = pd.read_csv(os.path.join(model_dir, "log.csv"))
# rreturn = df['rreturn_mean']

# plt.subplot(2, 1, 2)
# plt.plot(range(len(rreturn)), rreturn, color='black')

# # Color every point based on its color from the last plot
# for i in range(len(rreturn)):
#     plt.scatter(i, rreturn[i], color=env_indicator[i])

# plt.xlabel("Update")
# plt.ylabel("Rreturn")
# plt.title("Rreturn")


# # Display the plots
# plt.tight_layout()
# plt.show()

            

        # plt.plot(logs["return_per_episode"])
        # plt.xlabel("Episode")
        # plt.ylabel("Return")
        # plt.title("Return per Episode")
        # plt.show()
        # df = pd.read_csv(os.path.join(model_dir, "log.csv"))

        # # Extract the return mean column and plot it as a line graph
        # df['return_mean'].plot()

        # # Set the plot title and axis labels
        # plt.title('Return Mean')
        # plt.xlabel('Update')
        # plt.ylabel('Return Mean')

        # # Display the plot
        # plt.show()
