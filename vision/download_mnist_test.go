package vision

import (
	"io/ioutil"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDownloadMNIST(t *testing.T) {
	dir, e := ioutil.TempDir("", "gotorch.vision.download_mnist_test")
	assert.NoError(t, e)
	defer os.RemoveAll(dir)

	assert.NoError(t, downloadMNIST(dir))
}
