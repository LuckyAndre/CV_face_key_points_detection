import torch
from torch import nn
from torchvision import models

from .utils import decode_sequence, abc

# ПРОВЕРИЛ
class FeatureExtractor(nn.Module):

    def __init__(self, input_size=(64, 320), output_len=20):
        super(FeatureExtractor, self).__init__()

        h, w = input_size
        resnet = getattr(models, 'resnet18')(pretrained=True) # зачем так сложно? models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2]) # берем все кроме пулинга и полносвязного слоя

        self.pool = nn.AvgPool2d(kernel_size=(h // 32, 1)) # kernel_size = (2, 1), tensor [2 x 10] -> [1 x 10]
        self.proj = nn.Conv2d(w // 32, output_len, kernel_size=1) # Conv2d(10, 20, 1), перед этой сверткой выполняется изменение размерностей

        self.num_output_features = self.cnn[-1][-1].bn2.num_features

    def apply_projection(self, x):
        """Use convolution to increase width of a features.

        Args:
            - x: Tensor of features (shaped B x C x H x W).

        Returns:
            см. комментарий ниже
        """
        # B x 512 x 1 x 10
        x = x.permute(0, 3, 2, 1).contiguous() # этот permute нужен чтобы "10" перенести в размерность 1 (это позволит применить свертку и увеличить размер с 10 до 20)
        # B x 10 x 1 x 512
        x = self.proj(x) # cвертка для увеличения размера 10 -> 20
        # B x 20 x 1 x 512
        x = x.permute(0, 2, 3, 1).contiguous() # эта штука не нужна - см. комментарий ниже!
        """
        # B x 1 x 512 x 20
        Последний указнный permute на мой взгляд не нужен!
        Нам нужно тензор подавать в RNN, где требуется последовательность seq_len x B x H (т.е. 20 x B x 512)
        Поэтому правильнее здесь сразу привести к этой размерности!
        1) убираем x = x.permute(0, 2, 3, 1).contiguous()
        2) x = x.squeeze(2)
        3) x = x.permute(1, 0, 2).contiguous()
        """

        return x

    def forward(self, x):
        # Apply conv layers
        features = self.cnn(x)

        # Pool to make height == 1
        features = self.pool(features)

        # Apply projection to increase width
        features = self.apply_projection(features)

        return features


# ПРОВЕРИЛ
class SequencePredictor(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3, bidirectional=False):
        super(SequencePredictor, self).__init__()

        self.num_classes = num_classes
        self.rnn = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          dropout=dropout,
                          bidirectional=bidirectional)

        fc_in = hidden_size if not bidirectional else 2 * hidden_size
        self.fc = nn.Linear(in_features=fc_in,
                            out_features=num_classes)

    def _init_hidden(self, batch_size):
        """Initialize new tensor of zeroes for RNN hidden state.

        Args:
            - batch_size: Int size of batch

        Returns:
            Tensor of zeros shaped (num_layers * num_directions, batch, hidden_size).
        """
        num_directions = 2 if self.rnn.bidirectional else 1
        h = torch.zeros(self.rnn.num_layers * num_directions, batch_size, self.rnn.hidden_size)
        return h

    def _reshape_features(self, x):
        """Change dimensions of x to fit RNN expected input.

        Args:
            - x: Tensor x shaped (B x (C=1) x H x W).

        Returns:
            New tensor shaped (W x B x H).
        """
        # B x (C=1) x H x W
        x = x.squeeze(1)
        # B x H x W
        x = x.permute(2, 0, 1)
        # W x B x H

        return x

    def forward(self, x):
        x = self._reshape_features(x)
        batch_size = x.size(1)
        h_0 = self._init_hidden(batch_size)
        h_0 = h_0.to(x.device)
        x, h = self.rnn(x, h_0)
        x = self.fc(x)
        return x

# ПРОВЕРИЛ
class CRNN(nn.Module):

    def __init__(self, alphabet=abc,
                 cnn_input_size=(64, 320), cnn_output_len=20,
                 rnn_hidden_size=128, rnn_num_layers=1, rnn_dropout=0.0, rnn_bidirectional=False):
        super(CRNN, self).__init__()
        self.alphabet = alphabet
        self.features_extractor = FeatureExtractor(input_size=cnn_input_size, output_len=cnn_output_len)
        self.sequence_predictor = SequencePredictor(input_size=self.features_extractor.num_output_features,
                                                    hidden_size=rnn_hidden_size, num_layers=rnn_num_layers,
                                                    num_classes=len(alphabet) + 1, dropout=rnn_dropout,
                                                    bidirectional=rnn_bidirectional)

    def forward(self, x, decode=False):
        features = self.features_extractor(x)
        sequence = self.sequence_predictor(features)
        if decode:
            sequence = decode_sequence(sequence, self.alphabet)
        return sequence


def get_model():
    return CRNN()
