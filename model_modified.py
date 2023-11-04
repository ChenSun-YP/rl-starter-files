import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac



# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False,concatenate_context=False):
        super().__init__()

        # Decide which components are enabled
        # add context to obs_space and its a size 4 latent embedding    
        obs_space["context"] = 4
        self.use_text = use_text
        self.use_memory = use_memory
        self.concatenate_context = concatenate_context
        self.use_context = True
        self.context_size=4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64
        print('IMAGE EMBEDDING SIZE',self.image_embedding_size,obs_space)
        # Resize image embedding
        self.embedding_size = self.semi_memory_size


         # Add context inference related parameters and layers
        self.context_inputs = torch.nn.Parameter(torch.ones(self.context_size, device=self.device)/self.context_size)
        print('CONTEXTINPUT',self.context_inputs)
        self.context_wts = [{"params": self.context_inputs}]
        self.context_criterion = torch.nn.CrossEntropyLoss()
        self.context_lr = 0.1
        self.context_optimizer = torch.optim.SGD(self.context_wts, lr=self.context_lr)
        # self.agent_wts = [{"params": self.recurrent_net.parameters()},
                        #   {"params": self.readout.parameters()}]
        # self.agent_criterion = torch.nn.CrossEntropyLoss()
        # self.agent_lr = 0.001
        # self.agent_optimizer = torch.optim.Adam(self.agent_wts, lr=self.agent_lr)

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        if self.use_context:
            self.context_embedding_size = 32   
            self.context_embedding = nn.Embedding(obs_space["context"], self.context_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.context_embedding_size, self.text_embedding_size, batch_first=True)

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size+self.context_size, 64),
            nn.Tanh(),# hidden
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),#hidden
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory,return_ctxt=False):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        # Get context representation
        # context_repr = self.get_context_repr(x)
        # Concatenate context representation to image embedding
        # x = torch.cat([x, context_repr], dim=1)


        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)
        if self.use_context:

            embed_context = self._get_embed_context(obs.context) #TODO
            embedding = torch.cat((embedding, embed_context[:, -1, :]), dim=1)
        multiplicative_context = True
        if not self.concatenate_context:
            if multiplicative_context:
                context_repr = self.context_inputs.tanh() # get the context latent
                context_repr = context_repr.unsqueeze(0).unsqueeze(1).expand(embedding.size())# reshape like hidden_states
                context_aware_hidden = embedding * context_repr # modulate hidden state by context (multiplicative interaction)
            else:
                context_repr = torch.sigmoid(self.context_inputs, ) # get the context latent
                context_repr = context_repr.unsqueeze(0).unsqueeze(1).expand(embedding.size())# reshape like hidden_states
                context_aware_hidden = embedding + context_repr # modulate hidden state by context (multiplicative interaction)
            # modulate hidden state by context
            # last_context_hidden = context_aware_hidden[:, -1, :] # get the last context modulated hidden state
        else:
            # last_context_hidden = embedding[:, -1, :]
            context_repr = self.context_inputs # get the context latent


        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)
        
        if return_ctxt: # return latent context in addition to output and hidden state
            # first reshape
            return  dist, value, memory, context_repr
        else:
            return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
    def _get_embed_context(self, context):
        _, hidden = self.text_rnn(self.context_embedding(context))
        return hidden[-1]
    


    
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