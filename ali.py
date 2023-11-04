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
  def __init__(self, obs_space, output_size, hidden_size=100, device=device):
    super().__init__()
    self.input_size = obs_space
    self.output_size = output_size
    self.hidden_size = hidden_size
    self.device = device
    self.concatenate_context = True

    n = obs_space["image"][0]
    m = obs_space["image"][1]
    self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

    
    if self.concatenate_context:
      self.context_size = 4
      self.recurrent_net = torch.nn.GRU(self.image_embedding_size +self.context_size, hidden_size, device=device) # input/receptors + reservoir
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
      

          def loss_stopped(losses, threshold=1e-2, patience=10):
            """
            Check if the loss has stopped decreasing.

            Parameters:
            - losses: A list of loss values.
            - threshold: The threshold for the relative change in loss.
            - patience: The number of iterations to wait before stopping.

            Returns:
            - True if the loss has stopped decreasing, False otherwise.
            """
            if len(losses) < patience:
              return False
            else:
              last_losses = losses[-patience:]
              relative_change = abs(last_losses[-1] - last_losses[0]) / last_losses[0]
              return relative_change < threshold

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