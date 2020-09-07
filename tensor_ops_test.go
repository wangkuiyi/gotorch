package gotorch_test

import (
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
	torch "github.com/wangkuiyi/gotorch"
	"github.com/x448/float16"
)

// >>> t = torch.tensor([[-0.5, -1.], [1., 0.5]])
// >>> s = torch.tensor([[-0.5, -1.], [1., 0.5]])
// >>> t+s
// tensor([[-1., -2.],
//         [ 2.,  1.]])
func TestArith(t *testing.T) {
	a := assert.New(t)
	r := torch.NewTensor([][]float32{{-0.5, -1}, {1, 0.5}})
	s := torch.NewTensor([][]float32{{-0.5, -1}, {1, 0.5}})
	q := r.Add(s, 1)
	expected := torch.NewTensor([][]float32{{-1, -2}, {2, 1}})
	a.True(torch.Equal(expected, q))

	q = r.Sub(s, 1)
	expected = torch.NewTensor([][]float32{{0, 0}, {0, 0}})
	a.True(torch.Equal(expected, q))

	q = r.Mul(s)
	expected = torch.NewTensor([][]float32{{0.25, 1}, {1, 0.25}})
	a.True(torch.Equal(expected, q))

	q = r.Div(s)
	expected = torch.NewTensor([][]float32{{1.0, 1.0}, {1.0, 1.0}})
	a.True(torch.Equal(expected, q))

}

func TestArithI(t *testing.T) {
	a := assert.New(t)

	x := torch.RandN([]int64{2, 3}, false)
	y := torch.RandN([]int64{2, 3}, false)
	z := torch.Add(x, y, 1)
	x.AddI(y, 1)
	a.True(torch.Equal(x, z))

	z = torch.Sub(x, y, 1)
	x.SubI(y, 1)
	a.True(torch.Equal(x, z))

	z = torch.Mul(x, y)
	x.MulI(y)
	a.True(torch.Equal(x, z))

	z = torch.Div(x, y)
	x.DivI(y)
	a.True(torch.Equal(x, z))
}

func TestPermute(t *testing.T) {
	a := assert.New(t)
	x := torch.NewTensor([][]float32{{3, 1}, {2, 4}})
	y := x.Permute([]int64{1, 0})
	expected := torch.NewTensor([][]float32{{3, 2}, {1, 4}})
	a.True(torch.Equal(expected, y))
}

func TestAllClose(t *testing.T) {
	a := assert.New(t)
	x := torch.NewTensor([]float32{8.31, 6.55, 1.39})
	y := torch.NewTensor([]float32{2.38, 3.12, 5.23})
	r := x.Mul(y)
	expected := torch.NewTensor([]float32{8.31 * 2.38, 6.55 * 3.12, 1.39 * 5.23})
	a.True(torch.AllClose(expected, r))
}

// >>> torch.eq(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
// tensor([[ True, False],
//         [False, True]])
func TestEq(t *testing.T) {
	a := torch.NewTensor([][]int16{{1, 2}, {3, 4}})
	b := torch.NewTensor([][]int16{{1, 3}, {2, 4}})
	c := torch.Eq(a, b)
	g := " 1  0\n 0  1\n[ CPUBoolType{2,2} ]"
	assert.Equal(t, g, c.String())
}

func TestTensorEq(t *testing.T) {
	a := torch.NewTensor([][]int16{{1, 2}, {3, 4}})
	b := torch.NewTensor([][]int16{{1, 3}, {2, 4}})
	c := a.Eq(b)
	g := " 1  0\n 0  1\n[ CPUBoolType{2,2} ]"
	assert.Equal(t, g, c.String())
}

func TestEqual(t *testing.T) {
	a := torch.NewTensor([]int64{1, 2})
	b := torch.NewTensor([]int64{1, 2})
	assert.True(t, torch.Equal(a, b))
}

// >>> s = torch.tensor([1,2])
// >>> t = torch.tensor([[1,2],[3,4]])
// >>> s.expand_as(t)
// tensor([[1, 2],
//         [1, 2]])
func TestExpandAs(t *testing.T) {
	a := torch.NewTensor([]int8{'a', 'b'})
	b := torch.NewTensor([][]int8{{1, 2}, {3, 4}})
	c := torch.ExpandAs(a, b)
	g := " 97  98\n 97  98\n[ CPUCharType{2,2} ]"
	assert.Equal(t, g, c.String())
}

func TestTensorExpandAs(t *testing.T) {
	a := torch.NewTensor([]int8{'a', 'b'})
	b := torch.NewTensor([][]int8{{1, 2}, {3, 4}})
	c := a.ExpandAs(b)
	g := " 97  98\n 97  98\n[ CPUCharType{2,2} ]"
	assert.Equal(t, g, c.String())
}

// >>> torch.flatten(torch.tensor([[-0.5, -1.], [1., 0.5]]), 0, 1)
// tensor([-0.5000, -1.0000,  1.0000,  0.5000])
func TestFlatten(t *testing.T) {
	r := torch.Flatten(torch.NewTensor([][]float32{{-0.5, -1}, {1, 0.5}}),
		0, 1)
	g := "-0.5000\n-1.0000\n 1.0000\n 0.5000\n[ CPUFloatType{4} ]"
	assert.Equal(t, g, r.String())
}

// >>> torch.nn.functional.leaky_relu(torch.tensor([[-0.5, -1.], [1., 0.5]]))
// tensor([[-0.0050, -0.0100],
//         [ 1.0000,  0.5000]])
func TestLeakyRelu(t *testing.T) {
	r := torch.LeakyRelu(torch.NewTensor([][]float32{{-0.5, -1}, {1, 0.5}}),
		0.01)
	g := "-0.0050 -0.0100\n 1.0000  0.5000\n[ CPUFloatType{2,2} ]"
	assert.Equal(t, g, r.String())
}

// >>> torch.nn.functional.log_softmax(torch.tensor([[-0.5, -1.], [1., 0.5]]), dim=1)
// tensor([[-0.4741, -0.9741],
//         [-0.4741, -0.9741]])
func TestLogSoftmax(t *testing.T) {
	r := torch.LogSoftmax(torch.NewTensor([][]float32{{-0.5, -1}, {1, 0.5}}),
		1)
	g := "-0.4741 -0.9741\n-0.4741 -0.9741\n[ CPUFloatType{2,2} ]"
	assert.Equal(t, g, r.String())
}

// >>> torch.mean(torch.tensor([[-0.5, -1.], [1., 0.5]]))
// tensor(0.)
func TestMean(t *testing.T) {
	r := torch.Mean(torch.NewTensor([][]float32{{-0.5, -1}, {1, 0.5}}))
	// BUG: The result should be 0.
	g := "0\n[ CPUFloatType{} ]"
	assert.Equal(t, g, r.String())
}

func TestTensorMean(t *testing.T) {
	x := torch.RandN([]int64{2, 3}, true)
	y := x.Mean()
	z := y.Item()
	assert.NotNil(t, z)
}

// >>> torch.relu(torch.tensor([[-0.5, -1.], [1., 0.5]]))
// tensor([[0.0000, 0.0000],
//         [1.0000, 0.5000]])
func TestRelu(t *testing.T) {
	r := torch.Relu(torch.NewTensor([][]float32{{-0.5, -1}, {1, 0.5}}))
	g := " 0.0000  0.0000\n 1.0000  0.5000\n[ CPUFloatType{2,2} ]"
	assert.Equal(t, g, r.String())
}

// >>> torch.sigmoid(torch.tensor([[-0.5, -1.], [1., 0.5]]))
// tensor([[0.3775, 0.2689],
//         [0.7311, 0.6225]])
func TestSigmoid(t *testing.T) {
	r := torch.Sigmoid(torch.NewTensor([][]float32{{-0.5, -1}, {1, 0.5}}))
	g := " 0.3775  0.2689\n 0.7311  0.6225\n[ CPUFloatType{2,2} ]"
	assert.Equal(t, g, r.String())
}

func TestStack(t *testing.T) {
	t1 := torch.RandN([]int64{2, 3}, false)
	t2 := torch.RandN([]int64{2, 3}, false)
	out := torch.Stack([]torch.Tensor{t1, t2}, 0)
	assert.Equal(t, []int64{2, 2, 3}, out.Shape())
}

func TestSqueeze(t *testing.T) {
	x := torch.RandN([]int64{2, 1, 2, 1, 2}, false)
	y := torch.Squeeze(x)
	assert.NotNil(t, y.T)
	z := torch.Squeeze(x, 1)
	assert.NotNil(t, z.T)
	assert.Panics(t, func() { torch.Squeeze(x, 1, 2) })
}

// >>> x = torch.tensor([[1,2,3,4],[4,5,6,7],[7,8,9,0]])
// >>> torch.sum(x)
// tensor(56)
// >>> torch.sum(x, 0)
// tensor([12, 15, 18, 11])
// >>> torch.sum(x, 1)
// tensor([10, 22, 24])
// >>> torch.sum(x, 0, True)
// tensor([[12, 15, 18, 11]])
// >>> torch.sum(x, 0, False)
// tensor([12, 15, 18, 11])
// >>> torch.sum(x, 1, True)
// tensor([[10],
//         [22],
//         [24]])
// >>> torch.sum(x, 1, False)
// tensor([10, 22, 24])
func TestSum(t *testing.T) {
	x := torch.NewTensor([][]float32{{1, 2, 3, 4}, {4, 5, 6, 7}, {7, 8, 9, 0}})

	assert.Equal(t, float32(56), x.Sum().Item().(float32))

	y := x.Sum(map[string]interface{}{"dim": 0})
	assert.True(t, torch.Equal(torch.NewTensor([]float32{12, 15, 18, 11}), y),
		"Got %v", y)

	y = x.Sum(map[string]interface{}{"dim": 1})
	assert.True(t, torch.Equal(torch.NewTensor([]float32{10, 22, 24}), y),
		"Got %v", y)

	y = x.Sum(map[string]interface{}{"dim": 0, "keepDim": true})
	assert.True(t, torch.Equal(torch.NewTensor([][]float32{{12, 15, 18, 11}}), y),
		"Got %v", y)

	y = x.Sum(map[string]interface{}{"dim": 0, "keepDim": false})
	assert.True(t, torch.Equal(torch.NewTensor([]float32{12, 15, 18, 11}), y),
		"Got %v", y)

	y = x.Sum(map[string]interface{}{"dim": 1, "keepDim": true})
	assert.True(t, torch.Equal(torch.NewTensor([][]float32{{10}, {22}, {24}}), y),
		"Got %v", y)

	y = x.Sum(map[string]interface{}{"dim": 1, "keepDim": false})
	assert.True(t, torch.Equal(torch.NewTensor([]float32{10, 22, 24}), y),
		"Got %v", y)
}

func TestTanh(t *testing.T) {
	a := torch.RandN([]int64{4}, false)
	b := torch.Tanh(a)
	assert.NotNil(t, b.T)
}

// >>> torch.topk(torch.tensor([[-0.5, -1.], [1., 0.5]]), 1, 1, True, True)
// torch.return_types.topk(
// values=tensor([[-0.5000],
//         [ 1.0000]]),
// indices=tensor([[0],
//         [0]]))
func TestTopK(t *testing.T) {
	r, i := torch.TopK(torch.NewTensor([][]float64{{-0.5, -1}, {1, 0.5}}),
		1, 1, true, true)
	gr := "-0.5000\n 1.0000\n[ CPUDoubleType{2,1} ]"
	gi := " 0\n 0\n[ CPULongType{2,1} ]"
	assert.Equal(t, gr, r.String())
	assert.Equal(t, gi, i.String())
}

// >>> torch.transpose(torch.tensor([[-0.5, -1.], [1., 0.5]]), 0, 1)
// tensor([[-0.5000,  1.0000],
//         [-1.0000,  0.5000]])
func TestTranspose(t *testing.T) {
	x := torch.NewTensor([][]float32{{-0.5, -1}, {1, 0.5}})
	g := "-0.5000  1.0000\n-1.0000  0.5000\n[ CPUFloatType{2,2} ]"
	assert.Equal(t, g, x.Transpose(0, 1).String())
}

// >>> x = torch.randn(4, 4)
// >>> x.size()
// torch.Size([4, 4])
// >>> y = x.view(16)
// >>> y.size()
// torch.Size([16])
// >>> z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
// >>> z.size()
// torch.Size([2, 8])

// >>> a = torch.randn(1, 2, 3, 4)
// >>> a.size()
// torch.Size([1, 2, 3, 4])
// >>> b = a.transpose(1, 2)  # Swaps 2nd and 3rd dimension
// >>> b.size()
// torch.Size([1, 3, 2, 4])
// >>> c = a.view(1, 3, 2, 4)  # Does not change tensor layout in memory
// >>> c.size()
// torch.Size([1, 3, 2, 4])
// >>> torch.equal(b, c)
// False
func TestTensorView(t *testing.T) {
	x := torch.Empty([]int64{4, 4}, false)
	y := x.View(16)
	assert.Equal(t, []int64{16}, y.Shape())
	z := x.View(-1, 8)
	assert.Equal(t, []int64{2, 8}, z.Shape())
	a := torch.RandN([]int64{1, 2, 3, 4}, false)
	b := a.Transpose(1, 2)
	c := a.View(1, 3, 2, 4)
	assert.False(t, torch.Equal(b, c))
}

func TestArgmin(t *testing.T) {
	x := torch.NewTensor([][]float32{{1, 2}, {3, 4}})
	assert.Equal(t, "0\n[ CPULongType{} ]", x.Argmin().String())

	x = torch.NewTensor([][]float32{{4, 3}, {2, 1}})
	assert.Equal(t, "3\n[ CPULongType{} ]", x.Argmin().String())

	// x = torch.tensor([[3,4],[2,1]]
	x = torch.NewTensor([][]float32{{3, 4}, {2, 1}})
	// x.argmin(0)
	assert.Equal(t, " 1\n 1\n[ CPULongType{2} ]", x.Argmin(0).String())
	// x.argmin(1)
	assert.Equal(t, " 0\n 1\n[ CPULongType{2} ]", x.Argmin(1).String())
	// x.argmin(0, True)
	assert.Equal(t, " 1  1\n[ CPULongType{1,2} ]", x.Argmin(0, true).String())
	// x.argmin(1, True)
	assert.Equal(t, " 0\n 1\n[ CPULongType{2,1} ]", x.Argmin(1, true).String())

	assert.Panics(t, func() { x.Argmin(1.0 /* must be int*/, true) })
	assert.Panics(t, func() { x.Argmin(1, 1.0 /*must be bool*/) })
}

func TestArgmax(t *testing.T) {
	x := torch.NewTensor([][]float32{{1, 2}, {3, 4}})
	assert.Equal(t, "3\n[ CPULongType{} ]", x.Argmax().String())

	x = torch.NewTensor([][]float32{{4, 3}, {2, 1}})
	assert.Equal(t, "0\n[ CPULongType{} ]", x.Argmax().String())

	// x = torch.tensor([[3,4],[2,1]]
	x = torch.NewTensor([][]float32{{3, 4}, {2, 1}})
	// x.argmax(0)
	assert.Equal(t, " 0\n 0\n[ CPULongType{2} ]", x.Argmax(0).String())
	// x.argmax(1)
	assert.Equal(t, " 1\n 0\n[ CPULongType{2} ]", x.Argmax(1).String())
	// x.argmax(0, True)
	assert.Equal(t, " 0  0\n[ CPULongType{1,2} ]", x.Argmax(0, true).String())
	// x.argmax(1, True)
	assert.Equal(t, " 1\n 0\n[ CPULongType{2,1} ]", x.Argmax(1, true).String())
}

func f16(x float32) uint16 {
	return float16.Fromfloat32(x).Bits()
}

func TestItem(t *testing.T) {
	x := torch.NewTensor([]byte{1})
	y := x.Item()
	assert.Equal(t, byte(1), y)
	assert.NotEqual(t, int8(1), y)
	assert.Equal(t, reflect.TypeOf(y).Kind(), reflect.Uint8)

	x = torch.NewTensor([]bool{true})
	y = x.Item()
	assert.Equal(t, true, y)
	assert.Equal(t, reflect.TypeOf(y).Kind(), reflect.Bool)

	x = torch.NewTensor([]bool{false})
	y = x.Item()
	assert.Equal(t, false, y)
	assert.Equal(t, reflect.TypeOf(y).Kind(), reflect.Bool)

	x = torch.NewTensor([]int8{1})
	y = x.Item()
	assert.Equal(t, int8(1), y)
	assert.NotEqual(t, byte(1), y)
	assert.Equal(t, reflect.TypeOf(y).Kind(), reflect.Int8)

	x = torch.NewTensor([]int16{1})
	y = x.Item()
	assert.Equal(t, int16(1), y)
	assert.Equal(t, reflect.TypeOf(y).Kind(), reflect.Int16)

	x = torch.NewTensor([]int32{1})
	y = x.Item()
	assert.Equal(t, int32(1), y)
	assert.NotEqual(t, int64(1), y)
	assert.Equal(t, reflect.TypeOf(y).Kind(), reflect.Int32)

	x = torch.NewTensor([]int64{1})
	y = x.Item()
	assert.Equal(t, int64(1), y)
	assert.NotEqual(t, int32(1), y)
	assert.Equal(t, reflect.TypeOf(y).Kind(), reflect.Int64)

	x = torch.NewTensor([]int32{0x7FFF_FFFF})
	y = x.Item()
	assert.Equal(t, int32(0x7FFF_FFFF), y)

	x = torch.NewTensor([]int32{-0x8000_0000})
	y = x.Item()
	assert.Equal(t, int32(-0x8000_0000), y)

	// half
	x = torch.NewTensor([]uint16{f16(1)})
	y = x.Item()
	assert.Equal(t, float32(1), y)

	// max half
	x = torch.NewTensor([]uint16{f16(65504)})
	y = x.Item()
	assert.Equal(t, float32(65504), y)

	x = torch.NewTensor([]uint16{f16(0.25)})
	y = x.Item()
	assert.Equal(t, float32(0.25), y)

	x = torch.NewTensor([]float32{1.0})
	y = x.Item()
	assert.Equal(t, float32(1.0), y)
	assert.Equal(t, reflect.TypeOf(y).Kind(), reflect.Float32)

	x = torch.NewTensor([]float32{-1.0})
	y = x.Item()
	assert.Equal(t, float32(-1.0), y)

	x = torch.NewTensor([]float64{1.0})
	y = x.Item()
	assert.Equal(t, float64(1), y)
	assert.Equal(t, reflect.TypeOf(y).Kind(), reflect.Float64)

	x = torch.NewTensor([]float64{-1})
	y = x.Item()
	assert.Equal(t, float64(-1), y)
}

// >>> x = torch.tensor([[1,2,3,4],[4,5,6,7],[7,8,9,0]])
// >>> x
// tensor([[1, 2, 3, 4],
//         [4, 5, 6, 7],
//         [7, 8, 9, 0]])
// >>> idx = torch.tensor([0,2])
// >>> torch.index_select(x, 0, idx)
// tensor([[1, 2, 3, 4],
//         [7, 8, 9, 0]])
// >>> torch.index_select(x, 1, idx)
// tensor([[1, 3],
//         [4, 6],
//         [7, 9]])
func TestIndexSelect(t *testing.T) {
	x := torch.NewTensor([][]float32{{1, 2, 3, 4}, {4, 5, 6, 7}, {7, 8, 9, 0}})
	idx := torch.NewTensor([]int64{0, 2})
	assert.Equal(t, " 1  2  3  4\n 7  8  9  0\n[ CPUFloatType{2,4} ]",
		x.IndexSelect(0, idx).String())
	assert.Equal(t, " 1  3\n 4  6\n 7  9\n[ CPUFloatType{3,2} ]",
		x.IndexSelect(1, idx).String())
}
