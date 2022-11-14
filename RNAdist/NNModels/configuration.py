from dataclasses import dataclass
import math
from typing import Union
import torch
from functools import cached_property


@dataclass()
class ModelConfiguration:
    """The Model configuration for training a DISTAtteNCionE Network.

    Args:
        model_checkpoint (str): Path to the model output file.
        model (Union[str, torch.nn.Module]): Either normal or small if type is str. If type is :class:`torch.nn.Module`
            make sure that the :func:`~forward` of it takes two arguments a pair_rep and a mask where pair_rep
            is supposed to have the following shape: `(B, N, N, 17)` and the output must be of shape `(B, N, N)`
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
        use_bppm (bool): Whether to use basepair probability matrix as a feature
        use_position (bool): Whether to use positional encoding as a feature
        use_nucleotide_encoding (bool): Whether to use nucleotide encoding as a feature
        random_shift (float): probability to apply random shift in position encoding.
        normalize_bpp (bool): Whether bpp matrix is min max normalized or not.
        training_stats (str): Path to the training stats tsv file.
    """
    model_checkpoint: str
    model: Union[str, torch.nn.Module] = "normal"
    masking: bool = True
    nr_layers: int = 1
    optimizer: str = "adamw"
    learning_rate: float = 0.01
    batch_size: int = 16
    validation_interval: int = 5
    sample: int = None
    patience: int = 20
    lr_step_size: int = None,
    momentum: float = 0
    weight_decay: float = 0
    gradient_accumulation: int = 1
    use_bppm: bool = True
    use_position: bool = True
    use_nucleotide_encoding: bool = True
    random_shift: float = None
    normalize_bpp: bool = False
    training_stats: str = None
    gradient_checkpointing: bool = False

    def __post_init__(self):
        """Check valid argument combinations

        """
        if self.sample:
            if self.sample % self.gradient_accumulation and not math.isinf(self.sample):
                raise ValueError(f"sample must be a multiple of gradient accumulation")
        if not any((self.use_position, self.use_bppm, self.use_nucleotide_encoding)):
            raise ValueError(f"One of use_position, use_bppm or use_nucleotide_encoding must be True")

    @property
    def input_dim(self):
        return len(self.indices)

    @cached_property
    def indices(self):
        indices = []
        if self.use_bppm:
            indices.append(0)
        if self.use_position:
            indices += range(5, 9)
            indices += range(13, 17)
        if self.use_nucleotide_encoding:
            indices += range(1, 5)
            indices += range(9, 13)
        indices = sorted(indices)
        if all((self.use_bppm, self.use_position, self.use_nucleotide_encoding)):
            assert indices == list(range(0, 17)), f"Not the expected indices\n" \
                                                  f"expected: {list(range(0, 17))}" \
                                                  f"but got: {indices}"
        return torch.tensor(indices)

    def __getitem__(self, item):
        return self.__dict__[item]

