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

// downloadMNIST downloads mnist dataset
func downloadMNIST(dir string) error {
	if e := downloadIfNotYet(images, dir); e != nil {
		return e
	}
	if e := downloadIfNotYet(labels, dir); e != nil {
		return e
	}
	return nil
}

// cacheDir returns the directory where mnist dataset is saved
func cacheDir(dir string) string {
	if dir != "" {
		return dir
	}

	u, e := user.Current()
	if e != nil {
		return "testdata/mnist"
	}
	return path.Join(u.HomeDir, ".cache/mnist")
}

func downloadIfNotYet(fn, dir string) error {
	f := path.Join(cacheDir(dir), fn)
	if !fileExists(f) {
		if e := download(site+fn+".gz", f, dir); e != nil {
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

func download(url, fn, dir string) error {
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

	if e := os.MkdirAll(cacheDir(dir), 0744); e != nil {
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
