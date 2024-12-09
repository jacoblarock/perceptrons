package main

import (
	"fmt"
	"math/rand"
)

func main() {
	// p := perceptron{0.5, []float32{1, 1, 1, 1}}
	var p perceptron
	p.learn_rate = 0.5
	p.weights = []float32{1, 1, 1}
	data := make([][]float32, 25000000)
	expects := make([]int, 25000000)
	for i := 0; i < 25000000; i++ {
		data[i] = make([]float32, 2)
		data[i][0] = float32(rand.Int() % 100 - 50)
		data[i][1] = float32(rand.Int() % 100 - 50)
		if data[i][1] < 0 {
			expects[i] = 0
		} else {
			expects[i] = 1
		}
	}
	fmt.Println(evaluate(p, data, expects))
	fmt.Println(p.weights)
	bulk_train(p, data, expects)
	fmt.Println(evaluate(p, data, expects))
	fmt.Println(p.weights)
	bulk_train(p, data, expects)
	fmt.Println(evaluate(p, data, expects))
	fmt.Println(p.weights)
	bulk_train(p, data, expects)
	fmt.Println(evaluate(p, data, expects))
	fmt.Println(p.weights)
}
