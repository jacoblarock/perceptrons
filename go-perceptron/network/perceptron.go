package main

import (
	"math"
	"math/rand"
	"fmt"
	"strings"
	"strconv"
)

type perceptron struct {
	learn_rate float64
	weights []float64
}

type layer struct { 
	perceps []perceptron
	outs []chan float64
}

func make_layer(in_size int, p_count int, learn_rate float64) layer {
	var l layer
	l.perceps = make([]perceptron, p_count)
	l.outs = make([]chan float64, p_count)
	for i := 0; i < p_count; i++ {
		var p perceptron
		p.weights = make([]float64, in_size)
		p.learn_rate = learn_rate
		l.perceps[i] = p
	}
	return l
}

func net_rand_init(net []layer) {
	for i := 0; i < len(net); i++ {
		for j := 0; j < len(net[i].perceps); j++ {
			for w := 0; w < len(net[i].perceps[j].weights); w++ {
				net[i].perceps[j].weights[w] = rand.Float64()
			}
		}
	}
}

func array_to_string(a []float64, delim string) string {
	return strings.Trim(strings.Replace(fmt.Sprint(a), " ", delim, -1), "[]")
}

func print_net(net []layer) {
	for i := 0; i < len(net); i++ {
		for j := 0; j < len(net[i].perceps); j++ {
			fmt.Print("Layer " + strconv.Itoa(i) + " Perceptron " + strconv.Itoa(j) + ":  ")
			fmt.Println(net[i].perceps[j].weights)
		}
	}
}

func relu(x float64) float64 {
	if x > 0 {
		return x
	} else {
		return 0
	}
}

func ddx_relu(x float64) float64 {
	if x > 0 {
		return 1
	} else {
		return 0
	}
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}
func ddx_sigmoid(x float64) float64 {
	return sigmoid(x) * (1 - sigmoid(x))
}

func mse(y []float64, pred []float64) float64 {
	sum := 0.0
	for i := 0; i < len(y); i++ {
		diff := y[i] - pred[i]
		sum += diff * diff
	}
	return sum / float64(len(y))
}

func eval_neuron(p perceptron, out chan float64, vals []float64) {
	var sum float64
	sum = 0
	for i := 0; i < len(vals); i++ {
		sum += p.weights[i] * vals[i]
	}
	out <- sigmoid(sum)
}

func eval_layer(l layer, vals []float64) {
	for i := 0; i < len(l.perceps); i++ {
		out := make(chan float64, 1)
		l.outs[i] = out
		go eval_neuron(l.perceps[i], l.outs[i], vals)
	}
}

func await_outputs(outs []chan float64) []float64 {
	out_floats := make([]float64, len(outs))
	for i := 0; i < len(outs); i++ {
		out_floats[i] = <-outs[i]
	}
	return out_floats
}

func eval_net(layers []layer, inputs []float64) []*[]float64 {
	outs := make([]*[]float64, len(layers))
	eval_layer(layers[0], inputs)
	for i := 1; i < len(layers); i++ {
		layer_ins := await_outputs(layers[i-1].outs)
		outs[i-1] = &layer_ins
		eval_layer(layers[i], layer_ins)
	}
	net_out := await_outputs(layers[len(layers)-1].outs)
	outs[len(outs)-1] = &net_out
	return outs
}

func output_weights(layers []layer, layer_index int, p_index int) []float64 {
	if layer_index == len(layers) - 1 {
		out := make([]float64, 1)
		out[0] = 1.0
		return out
	}
	out_count := len(layers[layer_index + 1].outs)
	out := make([]float64, out_count)
	for i := 0; i < out_count; i++ {
		out[i] = layers[layer_index + 1].perceps[i].weights[p_index]
	}
	return out
}

func unit_error(pred float64, out_weights []float64, next_errs []float64) float64 {
	if len(out_weights) != len(next_errs) {
		fmt.Print("weights and next error sizes should match!")
	}
	out := 0.0
	for i := 0; i < len(out_weights); i++ {
		out += ddx_sigmoid(pred) * out_weights[i] * next_errs[i]
	}
	return out
}

func weight_delta(learn_rate float64, unit_error float64, unit_out float64) float64 {
	return learn_rate * unit_error * unit_out
}

func update_layer_weights(layers []layer, layer_index int, next_errs []float64, pred []float64) {
	if layer_index >= len(layers) - 1 {
		return
	}
	for i := 0; i < len(layers[layer_index].perceps); i++ {
		out_weights := output_weights(layers, layer_index, i)
		unit_err := unit_error(pred[i], out_weights, next_errs)
		for j := 0; j < len(out_weights); j++ {
			delta := weight_delta(layers[layer_index].perceps[i].learn_rate, unit_err, pred[i])
			layers[layer_index + 1].perceps[j].weights[i] += delta
		}
	}
}

func update_net_weights(layers []layer, outs []*[]float64, y []float64) {
	last_err := make([]float64, len(*outs[len(outs) - 1]))
	for i := 0; i < len(y); i++ {
		last_err[i] = y[i] * y[i]
	}
	next_errs := &last_err
	for i := len(layers) - 1; i >= 0; i-- {
		update_layer_weights(layers, i, *next_errs, *outs[i])
		unit_errs := make([]float64, len(layers[i].perceps))
		for j := 0; j < len(layers[i].perceps); j++ {
			pred := (*outs[i])[j]
			out_weights := output_weights(layers, i, j)
			unit_errs[j] = unit_error(pred, out_weights, *next_errs)
		}
		next_errs = &unit_errs
	}
}
