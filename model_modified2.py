import numpy as np
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
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False,concatenate_context=True):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory
        self.concatenate_context = concatenate_context
        self.context_size = 4
        self.context_inputs = None
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

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size
        if self.concatenate_context:
            self.context_size = 4
        else:
            self.context_size = 0
        self.context_inputs=torch.nn.Parameter(torch.ones(self.context_size)/self.context_size)
        self.context_wts = [{"params": self.context_inputs}]
        self.context_criterion = torch.nn.CrossEntropyLoss()
        self.context_lr = 0.1


        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size+4, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size+4, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )



        # define a world model for identifing and perdicting the context
        self.world_model = WorldModel(image_shape=(7, 7, 3), latent_dim=4, hidden_size=64)
        self.internal_model = InternalModel(image_shape=(7, 7, 3), latent_dim=4, hidden_size=64)

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory,latent_z):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

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
        
        batch_size, _ = x.shape
        # print(embedding.shape) #torch.Size([16, 64])
        latent_z_batched = latent_z.unsqueeze(0).repeat(embedding.size(0), 1)

        embedding = torch.cat((embedding,latent_z_batched), dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory
    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
    





class WorldModel(nn.Module):
    def __init__(self, image_shape, latent_dim, hidden_size):
        super(WorldModel, self).__init__()
        # Image shape is HWC, but PyTorch expects CHW, so we permute
        self.channels, self.height, self.width = image_shape[2], image_shape[0], image_shape[1]
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size

        # Encoder: Convolutional layers to process input image
        self.encoder = nn.Sequential(
            nn.Conv2d(self.channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # this will reduce each spatial dimension by half
            # Add more layers if necessary
        )

        # Compute the flattened size of the feature maps after the convolutional layers
        self.flattened_size = self._get_flattened_size()

        # Fully connected layers for encoding
        self.fc_encoder = nn.Sequential(
            nn.Linear(self.flattened_size + self.latent_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
        )

        # Decoder: Fully connected layers to project back to feature map size
        self.fc_decoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.flattened_size),
            nn.ReLU()
        )

        # Decoder: Transposed Convolutional layers to construct the output image
        self.decoder = nn.Sequential(
            # Adjust kernel_size, stride, and padding to match the encoder's reduction
            nn.ConvTranspose2d(16, self.channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Assuming the input images are normalized between 0 and 1
        )

    def _get_flattened_size(self):
        # Pass a dummy tensor through the encoder to compute the size of the flattened feature maps
        with torch.no_grad():
            dummy_input = torch.randn(1, self.channels, self.height, self.width)
            dummy_output = self.encoder(dummy_input)
            return int(np.prod(dummy_output.size()[1:]))

    def forward(self, obs, latent_z):
        # Permute the obs image to be in the format [batch_size, channels, height, width]
        x = obs.image.permute(0, 3, 1, 2)
        # Pass the image through the encoder and flatten the output
        encoded_features = self.encoder(x)
        encoded_features = encoded_features.reshape(encoded_features.size(0), -1)

        # Concatenate the encoded features with the latent vector
        latent_z_batched = latent_z.unsqueeze(0).repeat(encoded_features.size(0), 1)
        combined = torch.cat((encoded_features, latent_z_batched), dim=1)

        # Pass through the fully connected encoder
        encoded = self.fc_encoder(combined)

        # Decode the encoded state into feature map size
        decoded_features = self.fc_decoder(encoded)
        decoded_features = decoded_features.reshape(-1, 16, self.height // 2, self.width // 2)

        # Pass through the transposed convolutional decoder
        decoded_image = self.decoder(decoded_features)

        # Ensure the output has the correct shape
        predicted_image = decoded_image
        if predicted_image.shape != obs.image.shape:
            # If not, resize it appropriately
            predicted_image = nn.functional.interpolate(predicted_image, size=(self.height, self.width), mode='nearest')

        # Permute it back to [batch_size, height, width, channels] before returning
        predicted_image = predicted_image.permute(0, 2, 3, 1)
        # print('predicted_image',predicted_image.shape)
        return predicted_image  # Return it in the same format as the input


class InternalModel(nn.Module):
    '''
    This is the internal model for the agent to predict the next latent_z
    '''
    def __init__(self, image_shape, latent_dim, hidden_size):
        super(InternalModel, self).__init__()
        # Image shape is HWC, but PyTorch expects CHW, so we permute
        self.channels, self.height, self.width = image_shape[2], image_shape[0], image_shape[1]
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size

        # Encoder: Convolutional layers to process input image
        self.encoder = nn.Sequential(
            nn.Conv2d(self.channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # this will reduce each spatial dimension by half
            # Add more layers if necessary
        )

        # Compute the flattened size of the feature maps after the convolutional layers
        self.flattened_size = self._get_flattened_size()

        # Fully connected layers for encoding
        self.fc_encoder = nn.Sequential(
            nn.Linear(self.flattened_size + self.latent_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
        )

        # Decoder: Fully connected layers to project back to feature map size
        self.fc_decoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.flattened_size),
            nn.ReLU()
        )
        # decoder, from feature map to latent_z
        self.fc_decoder_z = nn.Sequential(
            nn.Linear(self.hidden_size, self.latent_dim),
            nn.ReLU()
        )


    def _get_flattened_size(self):
        # Pass a dummy tensor through the encoder to compute the size of the flattened feature maps
        with torch.no_grad():
            dummy_input = torch.randn(1, self.channels, self.height, self.width)
            dummy_output = self.encoder(dummy_input)
            return int(np.prod(dummy_output.size()[1:]))

    def forward(self, obs, latent_z):
        # take in the image and current latent_z, output the next latent_z
        # Permute the obs image to be in the format [batch_size, channels, height, width]
        x = obs.image.permute(0, 3, 1, 2)
        # Pass the image through the encoder and flatten the output
        encoded_features = self.encoder(x)
        encoded_features = encoded_features.reshape(encoded_features.size(0), -1)

        # Concatenate the encoded features with the latent vector
        latent_z_batched = latent_z.unsqueeze(0).repeat(encoded_features.size(0), 1)
        combined = torch.cat((encoded_features, latent_z_batched), dim=1)

        # Pass through the fully connected encoder
        encoded = self.fc_encoder(combined)

        # Decode the encoded state into feature map size
        decoded_features = self.fc_decoder(encoded)
        decoded_features = decoded_features.reshape(-1, 16, self.height // 2, self.width // 2)

        decoded_z = self.fc_decoder_z(decoded_features)
        # print('decoded_image',decoded_image.shape)
        return decoded_z

        
