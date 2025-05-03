package main

import (
	"math/rand"
	"fmt"
	"strings"
	"strconv"
)

type perceptron struct {
	learn_rate float32
	weights []float32
}

type layer struct { 
	perceps []perceptron
	outs []chan float32
}

func make_layer(in_size int, p_count int, learn_rate float32) layer {
	var l layer
	l.perceps = make([]perceptron, p_count)
	l.outs = make([]chan float32, p_count)
	for i := 0; i < p_count; i++ {
		var p perceptron
		p.weights = make([]float32, in_size)
		p.learn_rate = learn_rate
		l.perceps[i] = p
	}
	return l
}

func net_rand_init(net []layer) {
	for i := 0; i < len(net); i++ {
		for j := 0; j < len(net[i].perceps); j++ {
			for w := 0; w < len(net[i].perceps[j].weights); w++ {
				net[i].perceps[j].weights[w] = rand.Float32()
			}
		}
	}
}

func array_to_string(a []float32, delim string) string {
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

func relu(val float32) float32 {
	if val > 0 {
		return val
	} else {
		return 0
	}
}

func eval_neuron(p perceptron, out chan float32, vals []float32) {
	var sum float32
	sum = 0
	for i := 0; i < len(vals); i++ {
		sum += p.weights[i] * vals[i]
	}
	out <- relu(sum)
}

func eval_layer(l layer, vals []float32) {
	for i := 0; i < len(l.perceps); i++ {
		go eval_neuron(l.perceps[i], l.outs[i], vals)
	}
}

func await_outputs(outs []chan float32) []float32 {
	out_floats := make([]float32, len(outs))
	for i := 0; i < len(outs); i++ {
		out_floats[i] = <-outs[i]
	}
	return out_floats
}

func eval_net(layers []layer, inputs []float32) []float32 {
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
