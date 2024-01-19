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

if __name__ == "__main__":
    args = parser.parse_args()
    env_swap = 0



    args.mem = args.recurrence > 1
    swap_seq = np.random.randint(0,100,size=int(args.frames/args.interval))

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
        for i in range(args.procs):
            envs.append(utils.make_env(args.env, args.seed + 10000 * i,t=args.interval,swap_seq=swap_seq))
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



            if args.algo == "ppo2":
                exps, logs1 = algo.collect_experiences()
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
                header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
                if args.algo == "ppo2":

                    data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"],logs["world_loss"]]
                    txt_logger.info(
                    "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}| wl {:.3f}"
                    .format(*data))
                    # txt_logger.info(algo.latent_z)
                    

                else:
                    data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]
                    txt_logger.info(
                    "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
                    .format(*data))
                tb_writer.add_scalar('env_swap', env_swap, num_frames)
                env_swap = 0



                header += ["return_" + key for key in return_per_episode.keys()]
                data += return_per_episode.values()

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
        swap_id = 0
        env_names = ["doorkey", "cross", "obstacle", "goodlava"]
        env_colors = {"doorkey": "red", "cross": "blue", "obstacle": "green", "goodlava": "yellow"}  # Example colors for each environment        swap_seq_data=[]

        for i in range(args.frames):
            # swap at every interval *16 frames

            if i % (args.interval * 16) == 0:
                swap_id = swap_id + 1
                
            env_name = env_names[swap_seq[swap_id] % 4]
            # print("num_frames:", i, "env:", env_name)

            env_id = env_names.index(env_name)
            tb_writer.add_scalar('Environment', env_id, i)


            

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
