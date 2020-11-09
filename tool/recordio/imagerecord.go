package recordio

import (
	"bytes"
	"encoding/gob"
)

// ImageRecord struct
type ImageRecord struct {
	Image []byte
	Label int
}

// Encode an ImageRecord to bytes
func (ir *ImageRecord) Encode() ([]byte, error) {
	buf := &bytes.Buffer{}
	err := gob.NewEncoder(buf).Encode(*ir)
	if err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// Decode an ImageRecord from bytes
func (ir *ImageRecord) Decode(buf []byte) error {
	err := gob.NewDecoder(bytes.NewBuffer(buf)).Decode(&ir)
	if err != nil {
		return err
	}
	return nil
}
