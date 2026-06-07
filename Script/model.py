import numpy as np
import torch
import torch.nn as nn
from functions import ReshapeForLSTM
import torch.nn.functional as F



class CNNEncoder(nn.Module):

    """
    Initializes CNN architecture and reshapes the output in preparation for LSTM encoding.
    """

    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=(0, 1), stride=(2, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Dropout2d(0.1),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=(0, 1), stride=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout2d(0.1),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=(0, 1), stride=(2, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Dropout2d(0.1),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(0, 1), stride=(2, 1)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Dropout2d(0.1),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=(0, 1), stride=(2, 1)),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Dropout2d(0.1),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=(0, 1), stride=(2, 1)),
            nn.BatchNorm2d(512),
            nn.ELU(),
            nn.Dropout2d(0.1),
        )

        self.reshape = ReshapeForLSTM()

    def forward(self, x):

        """
        Forward pass through the CNN followed by a concatenation of the feature maps.

        ----------

        Parameters:
            x (torch.float32) - Spectrogram(s) of shape [batch_size, 1, height, width].

        Returns:
            x (torch.float32) - Concatenated feature maps of shape [batch_size, sequence_length, hidden_size].
        """

        x = self.cnn(x)
        x = self.reshape(x)
        return x


class LSTMDecoder(nn.Module):

    """
    Initializes the decoder consisting of an LSTM layer followed by fully connected output layers.
    """

    def __init__(self, hidden_size, input_size=8, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=False)

        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size/4)),
            nn.GELU(),
            nn.Linear(int(hidden_size/4), 4)
        )

        self.start_freq_head = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size/4)),
            nn.GELU(),
            nn.Linear(int(hidden_size/4), 1)
        )

        self.end_freq_head = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size/4)),
            nn.GELU(),
            nn.Linear(int(hidden_size/4), 1)
        )

        self.start_time_head = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size/4)),
            nn.GELU(),
            nn.Linear(int(hidden_size/4), 1)
        )

        self.end_time_head = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size/4)),
            nn.GELU(),
            nn.Linear(int(hidden_size/4), 1)
        )

    def forward(self, target_sequence, encoder_hidden, encoder_cell, max_length=10, eos_token_id=3):

        """
        Teacher-forced forward pass through the decoder. At each step it is fed the ground-truth token
        from the previous position (step 0 uses a start token) and predicts the token at the current
        position. Runs the full max_length steps; padding and EOS are handled later by the masked loss.

        ----------

        Parameters:
            target_sequence (torch.float32) - target data of shape [batch_size, sequence_length, 8].
            encoder_hidden (torch.float32) - final encoder hidden state, [num_layers, batch_size, hidden_size].
            encoder_cell (torch.float32) - final encoder cell state, [num_layers, batch_size, hidden_size].
            max_length (int) - number of output steps to generate.
            eos_token_id (int) - index of the EOS flag within the 8-element token.

        Returns:
            final_outputs (dict) - stacked predictions per head, each [batch_size, max_length, ...].
            finished (torch.bool) - per-sample flag, whether an EOS was predicted.
        """

        batch_size = encoder_hidden.size(1)
        device = encoder_hidden.device

        current_input = torch.zeros(batch_size, 1, 8, device=device)
        current_input[:, :, eos_token_id] = 1.0

        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell

        all_outputs = {
            'classification': [],
            'start_freq': [],
            'end_freq': [],
            'start_time': [],
            'end_time': []
        }

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(max_length):

            if step >= target_sequence.size(1):
                break

            decoder_output, (decoder_hidden, decoder_cell) = self.lstm(
                current_input, (decoder_hidden, decoder_cell)
            )

            decoder_output_squeezed = decoder_output.squeeze(1)

            classification_output = self.classification_head(decoder_output_squeezed)
            start_freq_output = self.start_freq_head(decoder_output_squeezed)
            end_freq_output = self.end_freq_head(decoder_output_squeezed)
            start_time_output = self.start_time_head(decoder_output_squeezed)
            end_time_output = self.end_time_head(decoder_output_squeezed)

            all_outputs['classification'].append(classification_output)
            all_outputs['start_freq'].append(start_freq_output)
            all_outputs['end_freq'].append(end_freq_output)
            all_outputs['start_time'].append(start_time_output)
            all_outputs['end_time'].append(end_time_output)

            predicted_classes = torch.argmax(classification_output, dim=-1)
            is_eos = (predicted_classes == eos_token_id)
            finished = finished | is_eos

            current_input = target_sequence[:, step:step + 1, :] 

        final_outputs = {}
        for key in all_outputs:
            final_outputs[key] = torch.stack(all_outputs[key], dim=1)

        return final_outputs, finished

    @torch.no_grad()
    def generate(self, encoder_hidden, encoder_cell, max_length=10, eos_token_id=3):

        """
        Autoregressive inference. The decoder feeds its own prediction back in as the next input,
        forming each next input as an 8-element token [cw, lfm, hfm, eos, t_start, t_stop, f1, f2] from the head outputs. 
        Stops once every sample has predicted an EOS, or after max_length steps.

        ----------

        Parameters:
            encoder_hidden (torch.float32) - final encoder hidden state, [num_layers, batch_size, hidden_size].
            encoder_cell (torch.float32) - final encoder cell state, [num_layers, batch_size, hidden_size].
            max_length (int) - largest number of steps to generate.
            eos_token_id (int) - index of the EOS flag within the 8-element token.

        Returns:
            (dict) - the predictions at every step, one entry per output: the class scores of shape
                    [batch_size, steps, 4], and the four time/frequency values, each [batch_size, steps, 1],
                    where steps is the number generated (at most max_length).
            finished (torch.bool) - one flag per sample, True if it predicted an EOS.
        """

        batch_size = encoder_hidden.size(1)
        device = encoder_hidden.device

        current_input = torch.zeros(batch_size, 1, 8, device=device)
        current_input[:, :, eos_token_id] = 1.0  # start token

        h, c = encoder_hidden, encoder_cell
        outputs = {k: [] for k in
                   ['classification', 'start_freq', 'end_freq', 'start_time', 'end_time']}
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length):
            out, (h, c) = self.lstm(current_input, (h, c))
            out = out.squeeze(1)

            cls = self.classification_head(out)
            sf, ef = self.start_freq_head(out), self.end_freq_head(out)
            st, et = self.start_time_head(out), self.end_time_head(out)

            outputs['classification'].append(cls)
            outputs['start_freq'].append(sf)
            outputs['end_freq'].append(ef)
            outputs['start_time'].append(st)
            outputs['end_time'].append(et)

            pred_cls = torch.argmax(cls, dim=-1)
            finished = finished | (pred_cls == eos_token_id)
            if finished.all():
                break

            # build the next token 
            nxt = torch.zeros(batch_size, 1, 8, device=device)
            nxt[:, 0, :4] = F.one_hot(pred_cls, num_classes=4).float()
            nxt[:, 0, 4] = st.squeeze(-1)   # t_start
            nxt[:, 0, 5] = et.squeeze(-1)   # t_stop
            nxt[:, 0, 6] = sf.squeeze(-1)   # f1
            nxt[:, 0, 7] = ef.squeeze(-1)   # f2
            current_input = nxt

        return {k: torch.stack(v, dim=1) for k, v in outputs.items()}, finished


def concatenate_bidirectional_states(hidden, cell):

    """
    Concatenates the forward and backward states of a bidirectional LSTM into a single set suitable for initializing a unidirectional decoder.

    ----------

    Parameters:
        hidden (torch.float32) - hidden state of the bidirectional LSTM with shape (num_layers*2, batch, hidden_size).
        cell (torch.float32) - cell state of the bidirectional LSTM with shape (num_layers*2, batch, hidden_size).

    Returns:
        decoder_hidden (torch.float32) - concatenated hidden state with shape (num_layers, batch, 2*hidden_size).
        decoder_cell (torch.float32) - concatenated cell state with shape (num_layers, batch, 2*hidden_size).
    """

    forward_hidden = hidden[::2]
    backward_hidden = hidden[1::2]

    forward_cell = cell[::2]
    backward_cell = cell[1::2]

    decoder_hidden = torch.cat([forward_hidden, backward_hidden], dim=-1)
    decoder_cell = torch.cat([forward_cell, backward_cell], dim=-1)

    return decoder_hidden, decoder_cell


def forward_pass(source_data, target_data, encoder_cnn, encoder_lstm, decoder):

    """
    Complete forward pass through the encoder-decoder pipeline. Source data is passed through the CNN
    encoder to extract feature maps, which are then processed by a bidirectional LSTM encoder. The
    bidirectional encoder states are concatenated to match the decoder's expected input shape, and
    the decoder produces output predictions using the target data with teacher forcing.

    ----------

    Parameters:
        source_data (torch.float32) - input spectrograms with shape [batch_size, 1, height, width].
        target_data (torch.float32) - target sequence with shape [batch_size, sequence_length, 8].
        encoder_cnn (nn.Module) - CNN encoder module.
        encoder_lstm (nn.Module) - bidirectional LSTM encoder module.
        decoder (nn.Module) - LSTM decoder module with output heads.

    Returns:
        decoder_outputs (dict) - dictionary containing the decoder's classification and regression outputs.
    """

    # 1. Encode through CNN
    cnn_features = encoder_cnn(source_data)

    # 2. Encode through bidirectional LSTM
    encoder_output, (encoder_hidden, encoder_cell) = encoder_lstm(cnn_features)

    # 3. Concatenate bidirectional states for decoder
    decoder_hidden, decoder_cell = concatenate_bidirectional_states(encoder_hidden, encoder_cell)

    # 4. Decode - note: decoder expects (target_seq, hidden, cell)
    decoder_outputs, _ = decoder(target_data, decoder_hidden, decoder_cell)

    return decoder_outputs


def compute_masked_loss(predictions, targets):

    """
    Computes the combined masked loss for the encoder-decoder model.

    Classification (cross-entropy) is scored on every position up to and including the first EOS.
    The four regression outputs (start/end time and frequency) are scored only on the real pulse
    positions, since the EOS row has zero time/frequency values. Each component is masked and
    averaged over its valid positions, then combined into one weighted total.

    ----------

    Parameters:
        predictions (dict) - model outputs with keys 'classification', 'start_freq', 'end_freq',
                            'start_time', 'end_time'.
        targets (torch.float32) - target tensor of shape [batch_size, sequence_length, 8].

    Returns:
        total_loss (torch.float32) - weighted sum of the loss components.
        loss_dict (dict) - the individual unweighted components, for logging.
    """

    batch_size, seq_len, _ = targets.shape

    target_classes = targets[:, :, 3]             
    eos_flag = (target_classes > 0.5)

    class_mask = torch.zeros_like(eos_flag, dtype=torch.bool)
    for i in range(batch_size):
        first_eos = torch.where(eos_flag[i])[0][0]   
        class_mask[i, :first_eos + 1] = True

    reg_mask = class_mask & (~eos_flag)

    pred_seq_len = predictions['classification'].size(1) 

    class_idx    = torch.argmax(targets[:, :pred_seq_len, :4], dim=-1)
    start_time_t = targets[:, :pred_seq_len, 4]
    end_time_t   = targets[:, :pred_seq_len, 5]
    start_freq_t = targets[:, :pred_seq_len, 6]
    end_freq_t   = targets[:, :pred_seq_len, 7]

    class_mask = class_mask[:, :pred_seq_len].float()
    reg_mask   = reg_mask[:, :pred_seq_len].float()
    reg_denom  = reg_mask.sum().clamp(min=1.0) 

    classification_loss = F.cross_entropy(
        predictions['classification'].reshape(-1, 4),
        class_idx.reshape(-1),
        reduction='none'
    ).reshape(batch_size, pred_seq_len)

    start_freq_loss = F.mse_loss(predictions['start_freq'].squeeze(-1), start_freq_t, reduction='none')
    end_freq_loss   = F.mse_loss(predictions['end_freq'].squeeze(-1),   end_freq_t,   reduction='none')
    start_time_loss = F.mse_loss(predictions['start_time'].squeeze(-1), start_time_t, reduction='none')
    end_time_loss   = F.mse_loss(predictions['end_time'].squeeze(-1),   end_time_t,   reduction='none')

    classification_loss = (classification_loss * class_mask).sum() / class_mask.sum()
    start_freq_loss = (start_freq_loss * reg_mask).sum() / reg_denom
    end_freq_loss   = (end_freq_loss   * reg_mask).sum() / reg_denom
    start_time_loss = (start_time_loss * reg_mask).sum() / reg_denom
    end_time_loss   = (end_time_loss   * reg_mask).sum() / reg_denom

    reg_loss = start_freq_loss + end_freq_loss + start_time_loss + end_time_loss
    total_loss = classification_loss + 10.0 * reg_loss

    return total_loss, {
        'classification': classification_loss,
        'start_freq': start_freq_loss,
        'end_freq': end_freq_loss,
        'start_time': start_time_loss,
        'end_time': end_time_loss
    }