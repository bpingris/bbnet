Sources:
https://medium.com/datathings/neural-networks-and-backpropagation-explained-in-a-simple-way-f540a3611f5e
http://hmkcode.com/ai/backpropagation-step-by-step/
https://www.quora.com/What-happens-if-no-activation-function-is-used-in-the-hidden-layers-of-a-neural-network
https://github.com/zHaytam/FlexibleNeuralNetFromScratch/blob/master/neural_network.py
https://blog.zhaytam.com/2018/08/15/implement-neural-network-backpropagation/

Back Propagation (Gradient computation)
The backpropagation learning algorithm can be divided into two phases: propagation and weight update.
- from wiki - Backpropagatio.

## Phase 1: Propagation
Each propagation involves the following steps:
- Forward propagation of a training pattern's input through the neural network in order to generate the propagation's output activations.
- Backward propagation of the propagation's output activations through the neural network using the training pattern target in order to generate the deltas of all output and hidden neurons.

## Phase 2: Weight update
For each weight-synapse follow the following steps:
- Multiply its output delta and input activation to get the gradient of the weight.
- Subtract a ratio (percentage) of the gradient from the weight.
- This ratio (percentage) influences the speed and quality of learning; it is called the learning rate. The greater the ratio, the faster the neuron trains; the lower the ratio, the more accurate the training is. The sign of the gradient of a weight indicates where the error is increasing, this is why the weight must be updated in the opposite direction.
- Repeat phase 1 and 2 until the performance of the network is satisfactory.
