package nn

import (
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"
)

type layer1 struct {
	Module
}

func (l *layer1) Forward(x, y int) (int64, int64) {
	return int64(x), int64(y)
}

type layer2 struct {
	Module
}

func (l *layer2) Forward(x, y int64) string {
	return strconv.Itoa(int(x + y))
}

type layer3 struct {
	Module
}

func (l *layer3) Forward(sum string) int {
	r, e := strconv.Atoi(sum)
	if e != nil {
		return 42
	}
	return r
}

type model1 struct {
	Module
	module *SequentialModule
}

func (m *model1) Forward() int {
	return m.module.Forward(1, 2).(int)
}

func TestSequential(t *testing.T) {
	m := model1{module: Sequential(&layer1{}, &layer2{}, &layer3{})}
	assert.Equal(t, m.Forward(), 3)
}

type layer4 struct {
	Module
	module *SequentialModule
}

func (l *layer4) Forward(x, y int) string {
	return l.module.Forward(x, y).(string)
}

type model2 struct {
	Module
	module *SequentialModule
}

func (l *model2) Forward(x, y int) int {
	return l.module.Forward(x, y).(int)
}

func TestSequentialInSequential(t *testing.T) {
	m := model2{
		module: Sequential(
			&layer4{
				module: Sequential(&layer1{}, &layer2{}),
			},
			&layer3{}),
	}
	assert.Equal(t, m.Forward(1, 2), 3)
}
