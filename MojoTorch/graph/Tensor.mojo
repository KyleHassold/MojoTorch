### Imports ###

from random import rand, seed
from memory import memset_zero, memset
from math import trunc, mod, min, sqrt, log, cos
from sys.info import simdwidthof
from algorithm import parallelize, vectorize, vectorize_unroll

### Structs ###

@register_passable("trivial")
struct Tensor[T: DType = DType.float32]:
    var rank: Int
    var num_elements: Int
    var shape: Pointer[Int]
    var strides: Pointer[Int]
    var data: DTypePointer[T]
    var grad: DTypePointer[T]

    fn __init__(*_shape: Int) raises -> Self:
        let rank = len[Int](_shape)
        var num_elements = 1
        let shape = Pointer[Int]().alloc(rank)
        let strides = Pointer[Int]().alloc(rank)

        for i in range(rank-1, -1, -1):
            if _shape[i] <= 0:
                raise Error('Invalid Tensor shape dimension: Axis ' + String(i) + ' of size ' + String(_shape[i]))

            shape.store(i, _shape[i])
            strides.store(i, _shape[i+1] * strides[i+1] if i == rank-1 else 1)
            num_elements *= shape[i]

        let data = DTypePointer[T].alloc(num_elements)
        memset_zero[T](data, num_elements)

        let grad = DTypePointer[T].alloc(num_elements)
        memset_zero[T](grad, num_elements)

        return Tensor[T] {
            rank: rank,
            num_elements: num_elements,
            shape: shape,
            strides: strides,
            data: data,
            grad: grad,
        }

    fn __str__(self) -> String:
        var s: String = ""

        for i in range(self.num_elements):
            if mod[DType.int8, 1](i, self.shape.load(self.rank-1)) == 0:
                var num = self.num_elements
                var j = 0
                while mod[DType.uint8, 1](i, num) != 0:
                    num /= self.shape[j]
                    j += 1
                    
                for k in range(self.rank):
                    s += ']' if k < self.rank-j else ' '
                for _ in range(self.rank-j):
                    s += '\n'
                for k in range(self.rank):
                    s += '[' if k >= j else ' '

            let val = self.data.load(i)
            if val >= 0:
                s += ' '
            if T.is_integral():
                s += ' ' + String(val) + ' '
            elif T.is_floating_point():
                s += ' ' + float_to_string(val, 4) + ' '

        for _ in range(self.rank):
            s += ']'
        return s[self.rank*2:]

    @always_inline
    fn __getitem__(self, i: Int) -> SIMD[T, 1]:
        return self.data.load(i)

    @always_inline
    fn __getitem__(self, row: Int, col: Int) -> SIMD[T, 1]:
        return self.data.load(row*self.shape[1] + col)

    @always_inline
    fn __setitem__(self, i: Int, val: SIMD[T, 1]):
        self.data.store(i)

    @always_inline
    fn __setitem__(self, row: Int, col: Int, val: SIMD[T, 1]):
        self.data.store(row*self.shape[1] + col, val)

    @always_inline
    fn load[num: Int = 1](self, row: Int, col: Int) -> SIMD[T, num]:
        return self.data.simd_load[num](row*self.shape[1] + col)

    @always_inline
    fn load[num: Int = 1](self, i: Int) -> SIMD[T, num]:
        return self.data.simd_load[num](i)

    @always_inline
    fn store[num: Int = 1](self, row: Int, col: Int, val: SIMD[T, num]):
        self.data.simd_store[num](row*self.shape[1] + col, val)

    @always_inline
    fn store[num: Int = 1](self, i: Int, val: SIMD[T, num]):
        self.data.simd_store[num](i, val)

    @always_inline
    fn rand(self):
        seed()
        rand[T](self.data, self.num_elements)

    fn rand_range(self, min_val: SIMD[T, 1], max_val: SIMD[T, 1]):
        seed()
        rand[T](self.data, self.num_elements)

        @parameter
        fn set_range[nelts : Int](i : Int):
            self.data.simd_store[nelts](i, self.data.simd_load[nelts](i) * (max_val - min_val) + min_val)
        vectorize[simdwidthof[T](), set_range](self.num_elements)

    fn rand_norm(self, mu: SIMD[T, 1], std: SIMD[T, 1]):
        seed()
        let pi = 3.14159265358979
        let u1 = DTypePointer[T].alloc(self.num_elements) 
        rand[T](u1, self.num_elements)
        rand[T](self.data, self.num_elements)

        @parameter
        fn norm[nelts : Int](i : Int):
            let z = sqrt(-2.0 * log(u1.simd_load[nelts](i))) * cos(2.0 * pi * self.data.simd_load[nelts](i))
            self.data.simd_store[nelts](i, z * std + mu)
        vectorize[simdwidthof[T](), norm](self.num_elements)

    fn __mul__(self, B: Self) raises -> Self:
        var C = Self(self.shape[0], B.shape[1])
        alias nelts = simdwidthof[T]()

        @parameter
        fn calc_row(m: Int):
            for k in range(self.shape[1]):
                @parameter
                fn dot[nelts : Int](n : Int):
                    C.store[nelts](m,n, C.load[nelts](m,n) + self[m,k] * B.load[nelts](k,n))
                vectorize[nelts, dot](C.shape[1])
        parallelize[calc_row](C.shape[0], C.shape[0])

        return C

    fn __add__(self, B: Self) raises -> Self:
        var C = self
        alias nelts = simdwidthof[T]()

        @parameter
        fn add[nelts : Int](i : Int):
            C.store[nelts](i, self.load[nelts](i) + B.load[nelts](i))
        vectorize[nelts, add](C.shape[1])

        return C

### Helper Functions ###

fn float_to_string[T: DType](val: SIMD[T, 1], precision: Int = 4) -> String:
    let s = String(val)
    var s2: String = ''
    for i in range(len(s)):
        if s[i] == '.':
            return s2 + s[i:min(i+precision+1, len(s))]
        s2 += s[i]
    return s2


### Test Code ###

fn main() raises:
    let test = Tensor[DType.float32](3, 2)
    print('Blank Tensor:\n' + test.__str__())
    print('Tensor Number of Elements:', test.num_elements)
    print('Tensor Rank:', test.rank)

    test.rand()
    print('Random Tensor:\n' + test.__str__())

    test.rand_range(-200, 100)
    print('Random Range Tensor:\n' + test.__str__())

    test.rand_norm(1, 2)
    print('Normal Random Tensor:\n' + test.__str__())

    let a = Tensor[DType.float16](3, 2)
    a.rand()
    print('Random Tensor A:\n' + a.__str__())

    let b = Tensor[DType.float16](2, 4)
    b.rand()
    print('Random Tensor B:\n' + b.__str__())

    let c = a * b
    print('Tensor A*B:\n' + c.__str__())

    try:
        let d = Tensor[DType.float16](-2, 2, -5)
        print('Tensor D:\n' + d.__str__())
    except e:
        print(e)
