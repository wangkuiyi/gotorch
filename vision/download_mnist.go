package vision

import (
	"compress/gzip"
	"io"
	"net/http"
	"os"
	"os/user"
	"path"
)

const (
	images = "train-images-idx3-ubyte"
	labels = "train-labels-idx1-ubyte"
	site   = "http://yann.lecun.com/exdb/mnist/"
)

// DownloadMNIST downloads mnist dataset
func DownloadMNIST() error {
	if e := downloadIfNotYet(images); e != nil {
		return e
	}
	if e := downloadIfNotYet(labels); e != nil {
		return e
	}
	return nil
}

// MNISTDir returns the directory where mnist dataset is saved
func MNISTDir() string {
	u, e := user.Current()
	if e != nil {
		return "testdata/mnist"
	}
	return path.Join(u.HomeDir, ".cache/mnist")
}

func downloadIfNotYet(fn string) error {
	f := path.Join(MNISTDir(), fn)
	if !fileExists(f) {
		if e := download(site+fn+".gz", f); e != nil {
			return e
		}
	}
	return nil
}

func fileExists(fn string) bool {
	info, e := os.Stat(fn)
	if os.IsNotExist(e) {
		return false
	}
	return !info.IsDir()
}

func download(url, fn string) error {
	resp, e := http.Get(url)
	if e != nil {
		return e
	}
	defer resp.Body.Close()

	r, e := gzip.NewReader(resp.Body)
	if e != nil {
		return e
	}
	defer r.Close()

	if e := os.MkdirAll(MNISTDir(), 0744); e != nil {
		return e
	}

	f, e := os.Create(fn)
	if e != nil {
		return e
	}
	defer f.Close()

	_, e = io.Copy(f, r)
	if e != nil && e != io.EOF {
		return e
	}

	return nil
}
