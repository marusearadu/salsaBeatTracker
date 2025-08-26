import madmom
from madmom.ml.nn import NeuralNetwork as NN
# from madmom.models import BEATS_LSTM, BEATS_BLSTM

import numpy as np
from torch import nn

class MyMadmom(nn.Module):
    """
    PyTorch equivalent of madmom's RNN model with 3 LSTM layers + 1 feedforward layer
    """
    def __init__(self, madmomModel: NN):
        super(MyMadmom, self).__init__()
        # getting all the params
        self.ogModel    = madmomModel
        self.__getParams()
        weights = self.__extractMadmomWeights()
        # 3 LSTM layers + feedforward + activation
        
        self.lstm1 = nn.LSTM(
            input_size = self.inputSize,
            hidden_size = self.hiddenSize,
            bidirectional = self.bidir
        )
        self.lstm2 = nn.LSTM(
            input_size = self.hiddenSize * self.D,
            hidden_size = self.hiddenSize,
            bidirectional = self.bidir
        )
        self.lstm3 = nn.LSTM(
            input_size = self.hiddenSize * self.D,
            hidden_size = self.hiddenSize,
            bidirectional = self.bidir
        )
        self.ff         = nn.Linear(in_features = self.hiddenSize * self.D, out_features = self.outputSize)
        self.activation = nn.Sigmoid()
    

    def __getParams(self):
        inputSize     = None
        hiddenSize    = None
        outputSize    = None
        bidirectional = False

        for layer in self.ogModel.layers:
            if 'lstm' in type(layer).__name__.lower():
                # Get input size from first LSTM layer's input gate weights
                assert hasattr(layer, 'input_gate'), "Mono-layer doesn't have input-gate"
                if hasattr(layer.input_gate, 'weights'):
                    w = layer.input_gate.weights.shape
                    if inputSize is None:
                        inputSize  = w[0]
                        hiddenSize = w[1]
                    else:
                        assert hiddenSize == w[1], "Cannot proceed as all the hidden layers of the LSTM have varying hidden sizes"
            
            elif 'bidirectional' in type(layer).__name__.lower():
                bidirectional = True
                assert hasattr(layer, 'fwd_layer'), "Bi-Layer doesn't have forward layer"
                assert hasattr(layer, 'bwd_layer'), "Bi-Layer doesn't have backward layer"
                if hasattr(layer.fwd_layer, 'input_gate') and hasattr(layer.fwd_layer.input_gate, "weights"):
                    w = layer.fwd_layer.input_gate.weights.shape
                    if inputSize is None:
                        inputSize  = w[0]
                        hiddenSize = w[1]
                    else:
                        assert hiddenSize == w[1], "Cannot proceed as all the hidden layers of the LSTM have varying hidden sizes"

            elif "forward" in type(layer).__name__.lower():
                # Get output size from feedforward layer's weights
                if hasattr(layer, 'bias'):
                    outputSize = layer.bias.shape[0]

        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.bidir      = bidirectional
        self.D          = self.bidir + 1

    def __extractMadmomWeights(self):
        weights = {}
        gates = ["input_gate", "forget_gate", "cell", "output_gate"]
        for i, layer in enumerate(self.ogModel.layers):
            layerType = type(layer).__name__.lower()

            if "lstm" in layerType:
                for gateName in gates:
                    gate = getattr(layer, gateName)

                    weights[f"lstm{i}_{gateName}_weights"] = gate.weights
                    weights[f"lstm{i}_{gateName}_recurrent_weights"] = gate.recurrent_weights
                    weights[f"lstm{i}_{gateName}_bias"] = gate.bias

            elif "bidirectional" in layerType:
                for sublayerName in ["fwd_layer", "bwd_layer"]:
                    sublayer = getattr(layer, sublayerName)
                    for gateName in gates:
                        gate = getattr(sublayer, gateName)
                        
                        weights[f"bidir{i}_{sublayerName}_{gateName}_weights"] = gate.weights
                        weights[f"bidir{i}_{sublayerName}_{gateName}_recurrent_weights"] = gate.recurrent_weights
                        weights[f"bidir{i}_{sublayerName}_{gateName}_bias"] = gate.bias

            elif "forward" in layerType.lower():
                weights[f"forward{i}_weights"] = layer.weights
                weights[f"forward{i}_bias"] = layer.bias

        return weights

    def forward(self, x, reset = True):
        def _getZerosTensors():
            tens = torch.zeros((
                    self.D * self.lstm1.num_layers, 
                    x.size(1),
                    self.hiddenSize),
                device = x.device) if x.dim() == 3 else torch.zeros((self.D * self.lstm1.num_layers, self.hiddenSize), device = x.device)
            return (tens, tens)

        out, _ = self.lstm1(x,    _getZerosTensors()) if reset else self.lstm1(x)
        out, _ = self.lstm2(out,  _getZerosTensors()) if reset else self.lstm2(out)
        out, _ = self.lstm3(out,  _getZerosTensors()) if reset else self.lstm3(out)
        out    = self.ff(out)
        out    = self.activation(out)
        
        return out
