import torch.nn as nn
import torch.nn.functional as F

from modules.layer.PESymetry import PESymetryMean, RPESymetryMean
from modules.layer.ScaledSelfAttention import ScaledSelfAttention


class AttRPEAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(AttRPEAgent, self).__init__()
        self.args = args

        # TODO: Check what is input_shape
        self.rpe_embedding = RPESymetryMean(input_shape, args.rnn_hidden_dim)
        actor_pe_layers = [
            # nn.ELU(),
            ScaledSelfAttention(
                args.rnn_hidden_dim, q_dim=args.rnn_hidden_dim, v_dim=args.rnn_hidden_dim, bias=False),
            nn.ELU(),
            PESymetryMean(args.rnn_hidden_dim, args.rnn_hidden_dim),
            nn.ELU(),
            # PESymetryMean(in_dim * 5, in_dim * 5),
            # nn.LeakyReLU(),
            PESymetryMean(args.rnn_hidden_dim, args.n_actions)
            # nn.Softmax(dim=-1)
        ]
        self.actor = nn.Sequential(*actor_pe_layers)

        # self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        # self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        # self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        # return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
        output = self.rpe_embedding.diagonal.weight_hh.new(self.args.rnn_hidden_dim)
        return output

    def forward(self, inputs, hidden_state=None):
        b, a, e = inputs.size()
        # print(f"inputs.size(): {inputs.size()}")
        # print(f"hidden_state.size(): {hidden_state.size()}")
        
        # x = F.relu(self.fc1(inputs.view(-1, e)), inplace=True)
        if hidden_state is not None:
            hidden_state = hidden_state.reshape(-1, a, self.args.rnn_hidden_dim)
        h = self.rpe_embedding(inputs.view(-1, a, e), hidden_state)
        h = F.tanh(h)
        # print(f"h shape:{h.shape}")
        q = self.actor(h)
        # print(f"q shape:{q.shape}")
        return q.view(b, a, -1), h.view(b, a, -1)