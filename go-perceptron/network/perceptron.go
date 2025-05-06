package main

import (
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
	out <- relu(sum)
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

func eval_net(layers []layer, inputs []float64) []float64 {
	eval_layer(layers[0], inputs)
	if len(layers) == 1 {
		return await_outputs(layers[0].outs)
	}
	for i := 1; i < len(layers); i++ {
		layer_ins := await_outputs(layers[i-1].outs)
		eval_layer(layers[i], layer_ins)
	}
	return await_outputs(layers[len(layers)-1].outs)
}


