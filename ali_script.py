'''
Train the model on the dataset.

    Parameters:
    - model: The model to train.
    - dataset: A dataset made from a session of trials.
    - add_noise: If True, Gaussian noise is added to the inputs.
    - noise_std: Standard deviation of the zero-mean additive noise.
    - infer_switch: If True, uses loss "spike" detection to determine context switches.
    - env: An environment must be provided if infer_switch is False.
    - device: The device (MPS or CPU or GPU) to put the model and data on.

    Returns:
    - model: Trained model.
    - train_info: Model's outputs and hidden states on the dataset.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac
from ali import AgentNetwork, OnlineStatsEMA

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 10
output_size = 4
model = AgentNetwork(input_size, output_size)
model = model.to(device)
model.train()

# Set up for event boundary (i.e. context switch) inference
if infer_switch:
    # will use loss "spike" detection to infer context switch trial indices
    event_boundaries = [0] # needs to start non-empty
else:
    assert env is not None, "An `env` must be provided if `infer_switch` is False."
    # the true context switch trial indices are provided
    event_boundaries = list(np.array([1] + env.unwrapped.context_switch_trials) - 1)
# number of trials to look back at for context updates
buffer_time = 1
# will look for "spikes" in exponential moving average (EMA) of the loss
loss_stats = OnlineStatsEMA()
loss_averages, loss_variances = [], []
# will compute accuracy as EMA of hits
acc_stats = OnlineStatsEMA()
accuracies = []

# Things to log and plot
train_log = []
train_trials, train_losses, train_hits = [], [], []

# Decompose the dataset into its inputs and labels
X, y, r = dataset.tensors # X.shape := (num_trials, trial_time, input_dim)
X, y, r = X.to(device), y.to(device), r.to(device) # r := reinforcer

# Loop over data
for t in range(len(dataset)):
    # 'true' trial number (i.e. natural number)
    trial = t + 1

    # Get the data for the current trial t
    input, label = X[t].unsqueeze(0), y[t].unsqueeze(0) # add batch dimensions
    num_trials, trial_time, input_dim = input.shape # := (batch_size=1, seq_len, input_size)

    # Add noise to the input
    if add_noise:
        noise = noise_std * torch.randn_like(input)
        input += noise

    # Forward pass through the model
    output, hidden = model(input)

    # Compute loss and other metrics
    loss = model.agent_criterion(output, label)
    probs = torch.nn.functional.softmax(output, dim=-1)
    _, predicted = torch.max(probs, dim=-1) # predicted := action taken by model
    hit = (label == predicted).long() # correctness of action
    acc_stats.update(hit) # update online accuracy statistics
    accuracy = acc_stats.get_mean() # compute acccuracy as the EMA of trial hits
    context_latent = model.get_context_repr(input) # context representation
    context_aware_hidden = hidden #* context_latent # context modulated hidden state (multiplicative)

    # Multipy reinforcer by supervised signal to get reward
    if t + 1 < len(dataset):
        # this says that the reward at the next trial ...
        reward = r[t + 1] * hit  # ... depends on the correctness of action now
        X[t + 1, :, -1] = reward # reward at next trial
        if add_noise: # add noise to the reward
            X[t + 1, :, -1] += noise_std * torch.randn_like(X[t + 1, :, -1])

    # Update online statistics (mean and variance)
    _loss = loss.detach().item()
    loss_stats.update(_loss)
    loss_avg = loss_stats.get_mean()
    loss_var = loss_stats.get_variance()

    # Since in training mode, we we perform both context and agent weight updates
    if infer_switch: # using the inferred context switch points to update context and weights
        # 2 standard deviations above the running mean (outside 95.5% confident interval)
        threshold = loss_avg + (2 * np.sqrt(loss_var))
        # detect if spiked on current trial
        if (t - 1 != event_boundaries[-1]) and (_loss > threshold): # first condition because "spikes" usually occur back-to-back
            # append trial index to detected spikes
            event_boundaries.append(t)
        # after collecting `buffer_time` trials in the new context
        if (t - buffer_time) == event_boundaries[-1]:
            # trigger context inference if spikes
            model.update_context(y[t - buffer_time:t], # collected trials
                                    X[t - buffer_time:t]) # update context
        # after being in the new context for buffer_time trials
        elif (t - event_boundaries[-1]) < 0 or (t - event_boundaries[-1]) > buffer_time:
            # otherwise update agent weights
            model.update_weights(loss) # update weights

    else: # using the true context switch trials to update context and weights
        # if already buffer trials into a new context block, perform a context update
        if (t - buffer_time) in set(event_boundaries):
            model.update_context(y[t - buffer_time:t],
                                X[t - buffer_time:t]) # update context
        # otherwise, and if outside of the buffer period, update the agent weights
        elif (t - find_nearest(event_boundaries, t)) < 0 or (t - find_nearest(event_boundaries, t)) > buffer_time:
            model.update_weights(loss) # update weights