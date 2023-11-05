### Imports ###
from ..graph.Tensor import Tensor

### Structs ###

@register_passable("trivial")
struct Linear[T: DType = DType.float32]:
    var weights: Tensor[T]
    var biases: Tensor[T]

    fn __init__(in_features: Int, out_features: Int) raises -> Self:
        let weights = Tensor[T](in_features,out_features)
        weights.rand_norm(0, 1)
        let biases = Tensor[T](out_features)
        biases.rand_norm(0, 1)

        return Linear[T] {
            weights: weights,
            biases: biases
        }

    fn forward(self, inp: Tensor[T]) raises -> Tensor[T]:
        return inp * self.weights + self.biases

fn main() raises:
    let x = Linear(3, 2)