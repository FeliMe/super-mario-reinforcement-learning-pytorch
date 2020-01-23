# Reinforcement Learning on Super Mario Bros (NES) PyTorch

Mostly taken from [https://github.com/sebastianheinz/super-mario-reinforcement-learning](https://github.com/sebastianheinz/super-mario-reinforcement-learning)
and [https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html).

## Usage

Clone the repository and install the project and its dependencies.

    git clone https://github.com/FeliMe/super-mario-reinforcement-learning.git
    pip install -r requirements.txt

## Training

Adapt the config section in train.py and run the script

    python train.py

## Testing

Run the replay script to see your trained model run

    python replay.py MODEL_PATH