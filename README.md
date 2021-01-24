## TRADE Multi-Domain Self-Attention Dialogue State Tracking

## Model Architecture
The architecture of the model, which includes an utterance Bert-based encoder, a self-attention generator, and a slot gate, all of which are shared among domains.
The state generator will decode J times independently for all the possible (domain, slot) pairs.
The slot gate predicts whether the j-th (domain, slot) pair is triggered by the dialogue.
