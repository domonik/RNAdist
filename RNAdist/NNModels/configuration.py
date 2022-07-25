from dataclasses import dataclass
import math


@dataclass()
class ModelConfiguration:
    """The Model configuration for training a DISTAtteNCionE Network.

    Args:
        model_checkpoint (str): Path to the model output file.
        model (str): Either normal or small
        alpha (float): "weight for the first part of the combined loss."
             "The weight for the loss of elements i,j"
             "where j > i + round(min_loop_length/2). The other part"
             "will be weighted 1 - alpha"
        masking (bool): whether masking is applied during training
        nr_layers (int): How often the Pair Update module is stacked
        optimizer (str): Specifies the optimizer that is used. either adamw or sgd
        learning_rate (float): initial learning rate
        batch_size (int): batch size of a mini batch
        validation_interval (int): after how many epochs validation should be aplied
        sample (int): if your training set is large use this to only sample this nr of
            instances per epoch
        patience (int): patience of the training procedure
        lr_step_size (int): after how many epochs lr should drop by cur_lr * 0.1
            (Only applied if  optimizer is SGD)
        momentum (float): momentum used for optimization
        weight_decay (float): weight decay used in optimization
        gradient_accumulation (int): gradient accumulation mirrors larger batch size
            (batch_size * gradient_accumulation)
    """
    model_checkpoint: str
    model: str = "normal"
    alpha: float = 0.5
    masking: bool = True
    nr_layers: int = 1
    optimizer: str = "adamw"
    learning_rate: float = 0.01
    batch_size: int = 16
    validation_interval: int = 5
    sample: int = math.inf
    patience: int = 20
    lr_step_size: int = None,
    momentum: float = 0
    weight_decay: float = 0
    gradient_accumulation: int = 1

    def __post_init__(self):
        """Check valid argument combinations

        """
        if self.sample:
            if not self.sample % self.gradient_accumulation:
                raise ValueError(f"sample must be a multiple of gradient accumulation")

    def __getitem__(self, item):
        return self.__dict__[item]

