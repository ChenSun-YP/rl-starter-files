import numpy
import torch
import torch.nn.functional as F

from torch_ac.algos.base import BaseAlgo
import numpy as np
class PPO2Algo(BaseAlgo):
    

    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,
                 reshape_reward=None,infer_switch=True):
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.infer_switch = infer_switch


        assert self.batch_size % self.recurrence == 0
        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, eps=adam_eps)
        self.batch_num = 0
        self.loss_stats = OnlineStatsEMA()
        self.acc_stats = OnlineStatsEMA()
        self.world_model = self.acmodel.world_model
        self.internal_model = self.acmodel.internal_model
        latent_dim = self.world_model.latent_dim
        self.latent_z = torch.randn(latent_dim, requires_grad=True) # todo, this should not be here

        self.optimizer_world_model = torch.optim.Adam(self.world_model.parameters(), lr, eps=adam_eps)
        self.optimizer_internal_model = torch.optim.Adam(self.internal_model.parameters(), lr, eps=adam_eps)
        self.optimizer_laten_z= torch.optim.Adam([self.latent_z], lr, eps=adam_eps)
        #random disribution of latent Z

                

        


    def update_parameters(self, exps):
        # Collect experiences
        latent_z_history = []  # Initialize empty list to store history of latent_z


        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []
            log_world_losses = []
            log_internal_losses = []

            for inds in self._get_batches_starting_indexes():
                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0
                batch_world_loss = 0
                batch_internal_loss = 0

                # Initialize memory                

                if self.acmodel.recurrent:
                    memory = exps.memory[inds]

                for i in range(self.recurrence):
                    # Create a sub-batch of experience
                    latent_z_history.append(self.latent_z.clone())

                    sb = exps[inds + i]

                    # Compute loss
                    #print out all the parameters in obs and tell if it contains the latent Z
                    if self.acmodel.recurrent:
                        dist, value, memory = self.acmodel(sb.obs, memory * sb.mask,self.latent_z)
                    else:
                        dist, value = self.acmodel(sb.obs)

                    # Compute and update world model here using sb.obs and sb.next_obs
                    world_loss = self.update_world_model(sb.prev_obs, sb.obs)  # Assuming sb.next_obs is the ground truth next state
                    # interal_loss is updated via pervious latent_z,pervious obs and current latent_z
                    #if there is no latent z histroy, then we use the current latent z and if there is no latent z, we use the random latent z
                    if len(latent_z_history) == 1: # 
                        internal_loss = self.update_internal_model(sb.prev_obs,self.latent_z)
                    else:
                        internal_loss = self.update_internal_model(sb.prev_obs,latent_z_history[-2])  # Update internal model and latent Z
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()

                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss
                    # Update batch values

                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss
                    batch_world_loss = world_loss.item()
                    batch_internal_loss = internal_loss.item()

                    # Update memories for next epoch

                    if self.acmodel.recurrent and i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                    self.loss_stats.update(batch_loss.detach().item())  # Update online loss statistics
                    # if self.infer_switch:
                    #     buffer_time = 1
                    #     # Calculate loss variance and check for spike
                    #     loss_var = self.loss_stats.get_variance()
                    #     if batch_loss > self.loss_stats.get_mean() + 2 * np.sqrt(loss_var):
                    #         # Spike detected, update context 'Z'
                    #         print(self.acmodel.context_inputs)
                    #         if (t - buffer_time) == event_boundaries[-1]:
                    #         # trigger context inference if spikes
                    #             self.update_world_model() # update context
                    #         # self.context_optimizer.zero_grad()
                    #         # batch_loss.backward(retain_graph=True)
                    #         # self.context_optimizer.step()
                    #         print('Spike detected, context updated.')
                    #         print(self.acmodel.context_inputs)

                # Update batch values

                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence
                batch_world_loss /= self.recurrence
                batch_internal_loss /= self.recurrence

                # Update actor-critic

                self.optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters() if p.grad is not None) ** 0.5

                # grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters()) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step() #update

        

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm)
                log_world_losses.append(batch_world_loss)
                log_internal_losses.append(batch_internal_loss)
            # todo: update the latent z
            # self.update_internal_model(latent_z_history, exps) # Update internal model and latent Z

            latent_z_history = [] # Clear history of latent_z



        # Log some values

        logs = {
            "entropy": numpy.mean(log_entropies),
            "value": numpy.mean(log_values),
            "policy_loss": numpy.mean(log_policy_losses),
            "value_loss": numpy.mean(log_value_losses),
            "grad_norm": numpy.mean(log_grad_norms),
            "world_loss": numpy.mean(log_world_losses)
        }

        return logs
    def update_world_model(self, prev_obs, obs):
        '''
        Update the world model and latent Z.
        :param state: Current state.
        '''
        # Predict the next state using the current state and latent Z.
        # print(obs.image.shape)
        predicted_next_state = self.world_model(prev_obs, self.latent_z)

        # Calculate the loss between the predicted and actual next state.
        # print('predicted_next_state',predicted_next_state.image[0])
        # print('next_state_ground_truth',obs.imgae[0])

        loss = torch.nn.MSELoss()(predicted_next_state, obs.image)

        # Backpropagate the loss through the world model to update it and latent Z.
        self.optimizer_world_model.zero_grad()
        loss.backward()
        self.optimizer_world_model.step()
        #print out the change on the latent Z
        # print('latent_z',self.latent_z)
        return loss
    

    def update_internal_model(self, prev_obs,latent_z_history):
            """
            Updates the internal model using the history of latent_z and experiences.

            Parameters:
            latent_z_history: List of latent_z values from previous steps.
            exps: Collected experiences from the environment.
            """


            # Process the history of latent_z along with experiences
            # Here, we assume that the internal model takes the history of latent_z 
            # and possibly some parts of experiences as input and outputs a new latent_z

            predicted_next_z = self.internal_model(prev_obs, latent_z_history)
            latent_z_batched = self.latent_z.unsqueeze(0).repeat(predicted_next_z.size(0), 1)



            internal_model_loss = torch.nn.MSELoss()(predicted_next_z, latent_z_batched)


            # Backpropagate the loss and update the internal model
            self.optimizer_internal_model.zero_grad()
            internal_model_loss.backward()
            self.optimizer_internal_model.step()

            # Update the current latent_z with the output of the internal model
            # self.latent_z = predicted_next_z.detach()  # Detach to avoid unwanted gradient flow

            # Return the loss for logging or other purposes
            return internal_model_loss


    def update_latent_z(self, exps, n_iters=200, threshold=1e-2):
        """
        Update the latent vector 'latent_z' based on experiences.

        Parameters:
        - exps: Collected experiences from the environment.
        - n_iters: Number of iterations to perform the update.
        - threshold: Convergence threshold for early stopping.
        """
        losses = []
        for i in range(n_iters):
            # Compute policy and value outputs using current latent_z
            policy_loss, value_loss = self.compute_policy_value_losses(exps)
 

            # Combine losses if necessary
            total_loss = policy_loss + value_loss

            # Perform backpropagation and update latent_z
            self.optimizer_laten_z.zero_grad()
            total_loss.backward()
            self.optimizer_laten_z.step()

            losses.append(total_loss.item())

            # Check for convergence
            if loss_stopped(losses, threshold=threshold):
                break

        if not loss_stopped(losses, threshold=threshold):
            print("Latent_z update did not converge.")

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


    def get_ground_truth_latent_z(self):
        """
        Returns the ground truth latent_z for the current environment.
        """
        return self.current_env.ground_truth_z

    def _get_batches_starting_indexes(self):

        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.

        First, the indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`, shifted by `self.recurrence//2` one time in two for having
        more diverse batches. Then, the indexes are splited into the different batches.

        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch
        """

        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        # Shift starting indexes by self.recurrence//2 half the time
        if self.batch_num % 2 == 1:
            indexes = indexes[(indexes + self.recurrence) % self.num_frames_per_proc != 0]
            indexes += self.recurrence // 2
        self.batch_num += 1

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes


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


