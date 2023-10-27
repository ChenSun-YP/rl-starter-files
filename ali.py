# Create a new model class (an agent) that will perform the task
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac


import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AgentNetwork(torch.nn.Module):
  """
  A agent that uses a minimally gated GRU.
  """
  def __init__(self, input_size, output_size, hidden_size=100, device=device):
    super().__init__()
    self.input_size = input_size
    self.output_size = output_size
    self.hidden_size = hidden_size
    self.device = device
    self.concatenate_context = True

    n = obs_space["image"][0]
    m = obs_space["image"][1]
    self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

    
    if self.concatenate_context:
      self.context_size = 4
      self.recurrent_net = torch.nn.GRU(input_size+self.context_size, hidden_size, device=device) # input/receptors + reservoir
    else:
      self.context_size = hidden_size
      self.recurrent_net = torch.nn.GRU(input_size, hidden_size, device=device) # input/receptors + reservoir
   
    self.readout = torch.nn.Linear(hidden_size, output_size, device=device) # output/actuators
    self.recurrent_weights = self.recurrent_net.Wh.weight # aliasing the recurrent weight matrix for easy access

    # Variables for context inference
    self.register_parameter(name="context_inputs", param=torch.nn.Parameter(
        torch.ones(self.context_size, device=self.device)/self.context_size))
    self.context_wts = [{"params": self.context_inputs}]
    self.context_criterion = torch.nn.CrossEntropyLoss()
    self.context_lr = 0.1
    self.context_optimizer = torch.optim.SGD(self.context_wts, lr=self.context_lr)

    self.agent_wts = [{"params": self.recurrent_net.parameters()},
                      {"params": self.readout.parameters()}]
    self.agent_criterion = torch.nn.CrossEntropyLoss()
    self.agent_lr = 0.001
    self.agent_optimizer = torch.optim.Adam(self.agent_wts, lr=self.agent_lr)

  def get_recurrent_weights(self):
    return self.recurrent_weights.data.detach().clone()

  def get_context_inputs(self):
    return self.context_inputs.data.detach().clone()

  def forward(self, x, return_ctxt=False):
    """
    x: torch.Tensor, input vector.
    return_ctxt: bool, whether to return the inferred latent context.
    """
    # batch_size := num_trials & seq_len := trial_time
    batch_size, seq_len, input_size = x.shape
    assert input_size == self.input_size, "Input dimension does not match model input size."
    # extract hidden states
    if self.concatenate_context:
      expanded_context = self.context_inputs.expand(batch_size, seq_len, self.context_size)
      # print('shape of expanded_context: ', expanded_context.shape)
      x = torch.cat((x, expanded_context), dim=2)

      hidden_states = self.recurrent_net(x, return_all=True)
   
    else:
       hidden_states = self.recurrent_net(x, return_all=True) # (batch_size, seq_len, hidden_size)
    # compute context latents
    multiplicative_context = True
    if not self.concatenate_context:
      if multiplicative_context:
        context_repr = self.context_inputs.tanh() # get the context latent
        context_repr = context_repr.unsqueeze(0).unsqueeze(1).expand(hidden_states.size())# reshape like hidden_states
        context_aware_hidden = hidden_states * context_repr # modulate hidden state by context (multiplicative interaction)
      else:
        context_repr = torch.sigmoid(self.context_inputs, ) # get the context latent
        context_repr = context_repr.unsqueeze(0).unsqueeze(1).expand(hidden_states.size())# reshape like hidden_states
        context_aware_hidden = hidden_states + context_repr # modulate hidden state by context (multiplicative interaction)
    # modulate hidden state by context
      last_context_hidden = context_aware_hidden[:, -1, :] # get the last context modulated hidden state
    else:
      last_context_hidden = hidden_states[:, -1, :]
      context_repr = self.context_inputs # get the context latent
    # compute output (action)
    output = self.readout(last_context_hidden) # readout from the last context modulated state
    # return output and hidden state
    if return_ctxt: # return latent context in addition to output and hidden state
      # first reshape
      return output, hidden_states, context_repr
    # default to not returning latent context
    return output, hidden_states

  # Context-dependent methods
  @torch.no_grad()
  def get_context_repr(self, x):
      _, _, context_repr = self.forward(x=x, return_ctxt=True)
      return context_repr

  def update_weights(self, loss):
      self.agent_optimizer.zero_grad()
      loss.backward()
      self.agent_optimizer.step()

  def update_context(self, y, x, n_iters=200, threshold=1e-2):
      """
      x (n x seq_len * feature_dim)
      y (n x 1)           or (1, )
      ... where n is the batch size
      """
      # find context representation
      losses = []
      for i in range(n_iters):
          yhat, _ = self.forward(x)
          loss = self.context_criterion(yhat, y)
          self._update_context(loss)
          losses.append(loss.data)
          loss_stopped_ = loss_stopped(losses, threshold=threshold)
          if loss_stopped_:
              break
      if not loss_stopped_:
          print(f'Context loss did NOT converge after {n_iters} iterations.')
          pass

  def _update_context(self, loss):
      self.context_optimizer.zero_grad()
      loss.backward()
      self.context_optimizer.step()



#@title Helper class for computing online mean and variance
class OnlineStatsEMA:
    def __init__(self, alpha=0.02):

        self.alpha = alpha  # Smoothing factor
        self.ema_mean = None  # Initialize EMA mean as None to handle the first observation
        self.ema_variance = None  # Initialize EMA variance as None to handle the first observation

    def update(self, x):
        if self.ema_mean is None:
            self.ema_mean = x  # For the first observation, set EMA mean = x
            self.ema_variance = 0.0  # Initialize variance as 0 for the first observation
        else:
            # Update the EMA mean using EMA formula
            self.ema_mean = (1 - self.alpha) * self.ema_mean + self.alpha * x
            # Update the EMA variance using EMA formula
            variance = (x - self.ema_mean)**2
            if self.ema_variance is None:
                self.ema_variance = variance
            else:
                self.ema_variance = (1 - self.alpha) * self.ema_variance + self.alpha * variance

    def get_variance(self):
        return self.ema_variance

    def get_mean(self):
        return self.ema_mean

"""
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
"""
# Define the device to use
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Move the model to the device and set it to training mode
# env  = utils.make_env(args.env, args.seed + 10000 * i)
# model = AgentNetwork(input_size=env.observation_space.shape[0],
#                      output_size=env.action_space.n,
#                      hidden_size=100,
#                      device=device)
# model.train()

# # Set up for event boundary (i.e. context switch) inference
# if infer_switch:
#     # will use loss "spike" detection to infer context switch trial indices
#     event_boundaries = [0] # needs to start non-empty
# else:
#     assert env is not None, "An `env` must be provided if `infer_switch` is False."
#     # the true context switch trial indices are provided
#     event_boundaries = list(np.array([1] + env.unwrapped.context_switch_trials) - 1)
# # number of trials to look back at for context updates
# buffer_time = 1
# # will look for "spikes" in exponential moving average (EMA) of the loss
# loss_stats = OnlineStatsEMA()
# loss_averages, loss_variances = [], []
# # will compute accuracy as EMA of hits
# acc_stats = OnlineStatsEMA()
# accuracies = []

# # Things to log and plot
# train_log = []
# train_trials, train_losses, train_hits = [], [], []

# # Decompose the dataset into its inputs and labels
# X, y, r = dataset.tensors # X.shape := (num_trials, trial_time, input_dim)
# X, y, r = X.to(device), y.to(device), r.to(device) # r := reinforcer

# # # Loop over data
# # for t in range(len(dataset)):
# #     # 'true' trial number (i.e. natural number)
# #     trial = t + 1

# #     # Get the data for the current trial t
# #     input, label = X[t].unsqueeze(0), y[t].unsqueeze(0) # add batch dimensions
# #     num_trials, trial_time, input_dim = input.shape # := (batch_size=1, seq_len, input_size)

# #     # Add noise to the input
# #     if add_noise:
# #         noise = noise_std * torch.randn_like(input)
# #         input += noise

# #     # Forward pass through the model
# #     output, hidden = model(input)

# #     # Compute loss and other metrics
# #     loss = model.agent_criterion(output, label)
# #     probs = torch.nn.functional.softmax(output, dim=-1)
# #     _, predicted = torch.max(probs, dim=-1) # predicted := action taken by model
# #     hit = (label == predicted).long() # correctness of action
# #     acc_stats.update(hit) # update online accuracy statistics
# #     accuracy = acc_stats.get_mean() # compute acccuracy as the EMA of trial hits
# #     context_latent = model.get_context_repr(input) # context representation
# #     context_aware_hidden = hidden #* context_latent # context modulated hidden state (multiplicative)

# #     # Multipy reinforcer by supervised signal to get reward
# #     if t + 1 < len(dataset):
# #         # this says that the reward at the next trial ...
# #         reward = r[t + 1] * hit  # ... depends on the correctness of action now
# #         X[t + 1, :, -1] = reward # reward at next trial
# #         if add_noise: # add noise to the reward
# #             X[t + 1, :, -1] += noise_std * torch.randn_like(X[t + 1, :, -1])

# #     # Update online statistics (mean and variance)
# #     _loss = loss.detach().item()
# #     loss_stats.update(_loss)
# #     loss_avg = loss_stats.get_mean()
# #     loss_var = loss_stats.get_variance()

# #     # Since in training mode, we we perform both context and agent weight updates
# #     if infer_switch: # using the inferred context switch points to update context and weights
# #         # 2 standard deviations above the running mean (outside 95.5% confident interval)
# #         threshold = loss_avg + (2 * np.sqrt(loss_var))
# #         # detect if spiked on current trial
# #         if (t - 1 != event_boundaries[-1]) and (_loss > threshold): # first condition because "spikes" usually occur back-to-back
# #             # append trial index to detected spikes
# #             event_boundaries.append(t)
# #         # after collecting `buffer_time` trials in the new context
# #         if (t - buffer_time) == event_boundaries[-1]:
# #             # trigger context inference if spikes
# #             model.update_context(y[t - buffer_time:t], # collected trials
# #                                     X[t - buffer_time:t]) # update context
# #         # after being in the new context for buffer_time trials
# #         elif (t - event_boundaries[-1]) < 0 or (t - event_boundaries[-1]) > buffer_time:
# #             # otherwise update agent weights
# #             model.update_weights(loss) # update weights

# #     else: # using the true context switch trials to update context and weights
# #         # if already buffer trials into a new context block, perform a context update
# #         if (t - buffer_time) in set(event_boundaries):
# #             model.update_context(y[t - buffer_time:t],
# #                                 X[t - buffer_time:t]) # update context
# #         # otherwise, and if outside of the buffer period, update the agent weights
# #         elif (t - find_nearest(event_boundaries, t)) < 0 or (t - find_nearest(event_boundaries, t)) > buffer_time:
# #             model.update_weights(loss) # update weights



# # Create a blended environment
# env = BlendedEnv()

# # Loop over trials
# for t in range(num_trials):
#     # Get the observation for the current trial t
#     obs = env.reset()

#     # Add noise to the observation
#     if add_noise:
#         noise = noise_std * np.random.randn(*obs.shape)
#         obs += noise

#     # Forward pass through the model
#     output, hidden = model(obs)

#     # Compute loss and other metrics
#     loss = model.agent_criterion(output, env.current_task)
#     probs = torch.nn.functional.softmax(output, dim=-1)
#     _, predicted = torch.max(probs, dim=-1) # predicted := action taken by model
#     hit = (env.current_task == predicted).long() # correctness of action
#     acc_stats.update(hit) # update online accuracy statistics
#     accuracy = acc_stats.get_mean() # compute acccuracy as the EMA of trial hits
#     context_latent = model.get_context_repr(obs) # context representation
#     context_aware_hidden = hidden #* context_latent # context modulated hidden state (multiplicative)

#     # Multipy reinforcer by supervised signal to get reward
#     if t + 1 < num_trials:
#         # this says that the reward at the next trial ...
#         reward = r[t + 1] * hit  # ... depends on the correctness of action now
#         obs, _, _, _ = env.step(reward) # reward at next trial
#         if add_noise: # add noise to the reward
#             noise = noise_std * np.random.randn(*obs.shape)
#             obs += noise

#     # Update online statistics (mean and variance)
#     _loss = loss.detach().item()
#     loss_stats.update(_loss)
#     loss_avg = loss_stats.get_mean()
#     loss_var = loss_stats.get_variance()

#     # Since in training mode, we we perform both context and agent weight updates
#     if infer_switch: # using the inferred context switch points to update context and weights
#         # 2 standard deviations above the running mean (outside 95.5% confident interval)
#         threshold = loss_avg + (2 * np.sqrt(loss_var))
#         # detect if spiked on current trial
#         if (t - 1 != event_boundaries[-1]) and (_loss > threshold): # first condition because "spikes" usually occur back-to-back
#             # append trial index to detected spikes
#             event_boundaries.append(t)
#         # after collecting `buffer_time` trials in the new context
#         if (t - buffer_time) == event_boundaries[-1]:
#             # trigger context inference if spikes
#             model.update_context(env.current_task) # update context
#         # after being in the new context for buffer_time trials
#         elif (t - event_boundaries[-1]) < 0 or (t - event_boundaries[-1]) > buffer_time:
#             # otherwise update agent weights
#             model.update_weights(loss) # update weights

#     else: # using the true context switch trials to update context and weights
#         # if already buffer trials into a new context block, perform a context update
#         if (t - buffer_time) in set(event_boundaries):
#             model.update_context(env.current_task) # update context
#         # otherwise, and if outside of the buffer period, update the agent weights
#         elif (t - find_nearest(event_boundaries, t)) < 0 or (t - find_nearest(event_boundaries, t)) > buffer_time:
#             model.update_weights(loss) # update weights