- add comments
- implement DBNBeatDetector...?
- start fine-tuning

### Maybe later:
- tweak bias storage(?) - copilot says: 

*Bias storage / splitting: madmom exposes a single bias per gate. PyTorch stores two bias vectors per layer per direction, bias_ih and bias_hh. The effective bias is bias_ih + bias_hh. A safe mapping is set bias_ih = madmom_bias and bias_hh = 0 (or split them if you know the original split).*

*bias_ih := madmom_gate_bias_concatenated*  
*bias_hh := zeros_like(bias_ih) (or split if you want to distribute)*
- add typing annotations for functions (i.e., what does function x return?)
- madmom identifies beats from the start of the song - i need only from the moment annotated beats turns on; crop songs to match the interval [firstBeat - beatDist / 2; lastBeat + beatDist / 2]?
- 