import math

import paddle


@paddle.jit.to_static
def ShiftedSoftPlus(x: paddle.Tensor) -> paddle.Tensor:
    return paddle.nn.functional.softplus(x) - math.log(2.0)
