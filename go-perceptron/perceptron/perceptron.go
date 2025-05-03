package main

type perceptron struct {
	learn_rate float32
	weights []float32
}

func classify(p perceptron, vals []float32) int {
	var sum float32
	sum = 0
	for i := 0; i < len(vals); i++ {
		sum += p.weights[i] * vals[i]
	}
	if sum > 0 {
		return 1
	}
	return -1
}

func train(p perceptron, vals []float32, expect int) {
	var out perceptron
	out.learn_rate = p.learn_rate
	out.weights = p.weights
	if classify(p, vals) != expect {
		for i := 0; i < len(vals); i++ {
			out.weights[i] = out.weights[i] + out.learn_rate * float32(expect) * vals[i]
		}
	}
}

func bulk_train(p perceptron, val_list [][]float32, expects []int) {
	for i := 0; i < len(val_list); i++ {
		train(p, val_list[i], expects[i])
	}
}

func evaluate(p perceptron, val_list [][]float32, expects []int) int {
	wins := 0
	for i := 0; i < len(val_list); i++ {
		if classify(p, val_list[i]) == expects[i] {
			wins++
		}
	}
	return wins
}
