package variadic

// Has returns if key is specified in opts.
func Has(opts []map[string]interface{}, key string) bool {
	if len(opts) == 0 {
		return false
	}
	_, ok := opts[0][key]
	return ok
}

// Get returns the option specified in opts with key, or nil.
func Get(opts []map[string]interface{}, key string) interface{} {
	if len(opts) == 0 {
		return nil
	}
	return opts[0][key]
}

// Lookup returns the value and its existence of key.
func Lookup(opts []map[string]interface{}, key string) (interface{}, bool) {
	if len(opts) == 0 {
		return nil, false
	}
	v, ok := opts[0][key]
	return v, ok
}
