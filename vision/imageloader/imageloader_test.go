package imageloader

import (
	"archive/tar"
	"io/ioutil"
	"log"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	torch "github.com/wangkuiyi/gotorch"
	tgz "github.com/wangkuiyi/gotorch/tool/tgz"
	"github.com/wangkuiyi/gotorch/vision/transforms"
)

func synthesizeInvalidImageTgz(fn string) {
	b := []byte("I'm a text string")
	w, e := tgz.CreateFile(fn)
	if e != nil {
		log.Fatalf("Cannot create writer: %v", e)
	}
	hdr := &tar.Header{
		Name: "text_file",
		Mode: 0600,
		Size: int64(len(b)),
	}

	if e := w.WriteHeader(hdr); e != nil {
		log.Fatalf("Failed writing header: %v", e)
	}
	if _, e := w.Write(b); e != nil {
		log.Fatalf("Failed writing PNG encoding: %v", e)
	}
}
func TestImageTgzLoaderError(t *testing.T) {
	a := assert.New(t)
	f, e := ioutil.TempFile("", "input.tgz")
	a.NoError(e)
	synthesizeInvalidImageTgz(f.Name())
	vocab := map[string]int{"0": 0, "1": 1}
	trans := transforms.Compose(
		transforms.ToTensor(),
		transforms.Normalize([]float32{0.1307}, []float32{0.3081}),
	)
	loader, e := New(f.Name(), vocab, trans, 3, 3, 1, false, "gray")
	defer torch.FinishGC()
	a.NoError(e)
	a.False(loader.Scan())
	a.Error(loader.Err())
}
func TestImageTgzLoader(t *testing.T) {
	a := assert.New(t)
	d, e := ioutil.TempDir("", "gotorch_image_tgz_loader*")
	a.NoError(e)

	fn, e := tgz.SynthesizeTarball(d)
	a.NoError(e)
	expectedVocab := map[string]int{"0": 0, "1": 1}
	vocab, e := BuildLabelVocabularyFromTgz(fn)
	a.NoError(e)
	a.Equal(expectedVocab, vocab)
	trans := transforms.Compose(
		transforms.ToTensor(),
		transforms.Normalize([]float32{0.1307}, []float32{0.3081}),
	)
	loader, e := New(fn, vocab, trans, 3, 3, 1, false, "rgb")
	a.NoError(e)
	{
		// first iteration
		a.True(loader.Scan())
		data, label := loader.Minibatch()
		a.Equal([]int64{3, 3, 2, 2}, data.Shape())
		a.Equal([]int64{3}, label.Shape())
		a.Equal(" 1\n 0\n 1\n[ CPULongType{3} ]", label.String())
	}
	{
		// second iteration with minibatch size is 2
		a.True(loader.Scan())
		data, label := loader.Minibatch()
		a.Equal([]int64{2, 3, 2, 2}, data.Shape())
		a.Equal([]int64{2}, label.Shape())
		a.Equal(" 0\n 0\n[ CPULongType{2} ]", label.String())
	}
	// no more data at the third iteration
	a.False(loader.Scan())
	a.NoError(loader.Err())

	_, e = BuildLabelVocabularyFromTgz("no file")
	a.Error(e)
}

func TestCornerCase(t *testing.T) {
	a := assert.New(t)
	d, e := ioutil.TempDir("", "gotorch_image_tgz_loader*")
	a.NoError(e)

	fn, e := tgz.SynthesizeTarball(d)
	a.NoError(e)
	expectedVocab := map[string]int{"0": 0, "1": 1}
	vocab, e := BuildLabelVocabularyFromTgz(fn)
	a.NoError(e)
	a.Equal(expectedVocab, vocab)
	trans := transforms.Compose(
		transforms.ToTensor(),
		transforms.Normalize([]float32{0.1307}, []float32{0.3081}),
	)
	loader, e := New(fn, vocab, trans, 5, 3, 1, false, "rgb")
	a.NoError(e)
	a.NotPanics(func() {
		for loader.Scan() {
			data, label := loader.Minibatch()
			a.Equal([]int64{5, 3, 2, 2}, data.Shape())
			a.Equal([]int64{5}, label.Shape())
		}
	})
	a.NoError(loader.Err())

	a.Panics(func() { New(fn, vocab, trans, -1, 3, 1, false, "rgb") })
	a.Panics(func() { New(fn, vocab, trans, 0, 3, 1, false, "rgb") })

}

func TestImageTgzLoaderHeavy(t *testing.T) {
	// NOTE: you can download a subset of ImageNet dataset which contains 1k images from https://gotorch-ci.oss-cn-hongkong.aliyuncs.com/imagenet_train_shuffle_1k.tgz
	// and run `export GOTORCH_TEST_IMAGE_TGZ_PATH=/your/path/imagenet_train_shuffle_1k.tgz` to set the environment variable to run
	// this unit test.
	// If you want to generate a custom shuffled tarball, please go to https://github.com/wangkuiyi/gotorch/blob/develop/doc/shuffle_tarball.md
	if os.Getenv("GOTORCH_TEST_IMAGE_TGZ_PATH") == "" {
		t.Skip("No GOTORCH_TEST_IMAGE_TGZ_PATH from env, skip test")
	}
	trainFn := os.Getenv("GOTORCH_TEST_IMAGE_TGZ_PATH")
	mbSize := 32
	vocab, e := BuildLabelVocabularyFromTgz(trainFn)
	if e != nil {
		log.Fatal(e)
	}
	trans := transforms.Compose(
		transforms.RandomResizedCrop(224),
		transforms.RandomHorizontalFlip(0.5),
		transforms.ToTensor(),
		transforms.Normalize([]float32{0.485, 0.456, 0.406}, []float32{0.229, 0.224, 0.225}))

	loader, e := New(trainFn, vocab, trans, mbSize, mbSize, 1, false, "rgb")
	defer torch.FinishGC()
	if e != nil {
		log.Fatal(e)
	}
	startTime := time.Now()
	idx := 0
	for loader.Scan() {
		idx++
		loader.Minibatch()
		if idx%10 == 0 {
			throughput := float64(mbSize*10) / time.Since(startTime).Seconds()
			log.Printf("throughput: %f samples/secs", throughput)
			startTime = time.Now()
		}
	}
}

func TestSplitComposeByToTensor(t *testing.T) {
	a := assert.New(t)
	{
		trans := transforms.Compose(
			transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(0.5),
			transforms.ToTensor(),
			transforms.Normalize([]float32{0.485, 0.456, 0.406}, []float32{0.229, 0.224, 0.225}))
		trans1, trans2 := splitComposeByToTensor(trans)
		a.Equal(len(trans1.Transforms), 2)
		_, ok := trans1.Transforms[0].(*transforms.RandomResizedCropTransformer)
		a.True(ok)
		_, ok = trans1.Transforms[1].(*transforms.RandomHorizontalFlipTransformer)
		a.True(ok)
		a.Equal(len(trans2.Transforms), 2)
		_, ok = trans2.Transforms[0].(*transforms.ToTensorTransformer)
		a.True(ok)
		_, ok = trans2.Transforms[1].(*transforms.NormalizeTransformer)
		a.True(ok)
	}
	{
		trans := transforms.Compose(
			transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(0.5))
		trans1, trans2 := splitComposeByToTensor(trans)
		a.Equal(len(trans1.Transforms), 2)
		_, ok := trans1.Transforms[0].(*transforms.RandomResizedCropTransformer)
		a.True(ok)
		_, ok = trans1.Transforms[1].(*transforms.RandomHorizontalFlipTransformer)
		a.True(ok)
		a.Equal(len(trans2.Transforms), 0)
	}
}
