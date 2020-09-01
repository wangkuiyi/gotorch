package datasets

import (
	"io/ioutil"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/wangkuiyi/gotorch/tool/tgz"
	"github.com/wangkuiyi/gotorch/vision/transforms"
)

func TestImageTgzLoader(t *testing.T) {
	a := assert.New(t)
	d, e := ioutil.TempDir("", "gotorch_tarball_divide_test*")
	a.NoError(e)

	fn := tgz.SynthesizeTarball(t, d)
	expectedVocab := map[string]int64{"0": int64(0), "1": int64(1)}
	vocab, e := BuildLabelVocabularyFromTgz(fn)
	a.NoError(e)
	a.Equal(expectedVocab, vocab)

	trans := transforms.Compose(
		transforms.ToTensor(),
		transforms.Normalize([]float64{0.1307}, []float64{0.3081}),
	)
	loader, e := NewImageLoader(fn, vocab, trans, 3)
	a.NoError(e)
	{
		// first iteration
		a.True(loader.Scan())
		data, label := loader.Minibatch()
		a.Equal([]int64{3, 3, 2, 2}, data.Shape())
		a.Equal([]int64{3}, label.Shape())
	}
	{
		// second iteration with minibatch size is 2
		a.True(loader.Scan())
		data, label := loader.Minibatch()
		a.Equal([]int64{2, 3, 2, 2}, data.Shape())
		a.Equal([]int64{2}, label.Shape())
	}
	// no more data at the third iteration
	a.False(loader.Scan())
	a.NoError(loader.Err())
}
