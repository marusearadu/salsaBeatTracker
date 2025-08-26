import numpy as np
from pyparsing import Dict

import madmom
from madmom.ml.nn import NeuralNetwork as NN
# from madmom.models import BEATS_LSTM, BEATS_BLSTM

import torch
from torch import nn

class MyMadmom(nn.Module):
    """
    PyTorch equivalent of madmom's RNN model with 3 LSTM layers + 1 feedforward layer
    """
    def __init__(self, madmomModel: NN):
        super(MyMadmom, self).__init__()
        self.__setParams(madmomModel)

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

        weights = self.__extractMadmomWeights(madmomModel)
        self.__setWeights(weights)

    def __setParams(self, model: NN) -> None:
        inputSize     = None
        hiddenSize    = None
        outputSize    = None
        bidirectional = False

        for layer in model.layers:
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

    def __extractMadmomWeights(self, model: NN) -> Dict[str, np.ndarray]:
        weights = {}
        gates = ["input_gate", "forget_gate", "cell", "output_gate"]
        for i, layer in enumerate(model.layers):
            if self.bidir and hasattr(layer, "fwd_layer"):
                # bidirectional lstm layer
                for sublayerName in ["fwd_layer", "bwd_layer"]:
                    sublayer = getattr(layer, sublayerName)
                    for gateName in gates:
                        gate = getattr(sublayer, gateName)
                        
                        weights[f"bidir{i}_{sublayerName}_{gateName}_weights"] = gate.weights
                        weights[f"bidir{i}_{sublayerName}_{gateName}_recurrent_weights"] = gate.recurrent_weights
                        weights[f"bidir{i}_{sublayerName}_{gateName}_bias"] = gate.bias

            elif not self.bidir and hasattr(layer, gates[0]):
                # unidirectional lstm layer
                for gateName in gates:
                    gate = getattr(layer, gateName)

                    weights[f"lstm{i}_{gateName}_weights"] = gate.weights
                    weights[f"lstm{i}_{gateName}_recurrent_weights"] = gate.recurrent_weights
                    weights[f"lstm{i}_{gateName}_bias"] = gate.bias

            elif hasattr(layer, "weights") and hasattr(layer, "bias"):
                # feedforward layer
                weights[f"forward{i}_weights"] = layer.weights
                weights[f"forward{i}_bias"] = layer.bias

        return weights

    def __setWeights(self, weights: Dict[str, np.ndarray]):
        def __concatWeights(prefix: str):
            W = torch.from_numpy(np.concatenate([
                    weights[f"{prefix}_input_gate_weights"].T,
                    weights[f"{prefix}_forget_gate_weights"].T,
                    weights[f"{prefix}_cell_weights"].T,
                    weights[f"{prefix}_output_gate_weights"].T
                ])).float()
            R = torch.from_numpy(np.concatenate([
                weights[f"{prefix}_input_gate_recurrent_weights"].T,
                weights[f"{prefix}_forget_gate_recurrent_weights"].T,
                weights[f"{prefix}_cell_recurrent_weights"].T,
                weights[f"{prefix}_output_gate_recurrent_weights"].T
            ])).float()
            b = torch.from_numpy(np.concatenate([
                weights[f"{prefix}_input_gate_bias"].T,
                weights[f"{prefix}_forget_gate_bias"].T,
                weights[f"{prefix}_cell_bias"].T,
                weights[f"{prefix}_output_gate_bias"].T
            ])).float()

        def __setLSTMlayerWeights(lstmLayer: nn.LSTM, layerID: int):
            prefix = f"bidir{layerID}_fwd_layer" if self.bidir else f"lstm{layerID}"
            W, R, b = __concatWeights(prefix)

            lstmLayer.weight_hh_l0.data = R
            lstmLayer.weight_ih_l0.data = W
            lstmLayer.bias_ih_l0.data   = b
            lstmLayer.bias_hh_l0.data   = torch.zeros_like(b)

            if self.bidir:
                W, R, b = __concatWeights(f"bidir{layerID}_bwd_layer")

                lstmLayer.weight_hh_l0_reverse.data = R
                lstmLayer.weight_ih_l0_reverse.data = W
                lstmLayer.bias_ih_l0_reverse.data   = b
                lstmLayer.bias_hh_l0_reverse.data   = torch.zeros_like(b)

            return W, R, b

        __setLSTMlayerWeights(self.lstm1, 0)
        __setLSTMlayerWeights(self.lstm2, 1)
        __setLSTMlayerWeights(self.lstm3, 2)

        self.ff.weight.data = torch.from_numpy(self.weights["forward3_weights"].T).float()
        self.ff.bias.data   = torch.from_numpy(self.weights["forward3_bias"].T).float()

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
