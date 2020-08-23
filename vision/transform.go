package vision

// NormalizeTransformer corresponds to torchvision.transforms.html#Normalize. It
// implements Go interface gotorch/data.Transform.
type NormalizeTransformer struct {
	mean, stddev float64
}

// Normalize returns normalize transformer
func Normalize(mean float64, stddev float64) *NormalizeTransformer {
	return &NormalizeTransformer{mean, stddev}
}
