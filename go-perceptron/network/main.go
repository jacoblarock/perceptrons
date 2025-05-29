package main

import (
	"fmt"
)

func main() {
	net := make([]layer, 3)
	net[0] = make_layer(2, 2, 0.5)
	net[1] = make_layer(2, 2, 0.5)
	net[2] = make_layer(2, 1, 0.5)
	net_rand_init(net)
	print_net(net)
	y := []float64 {0.69}
	test := make([]float64, 2)
	for i := 0; i < len(test); i++ {
		test[i] = 0.1
	}
	res := eval_net(net, test)
	for i := 0; i < len(res); i++ {
		fmt.Println(*res[i])
	}
	for i := 0; i < 100; i++ {
		update_net_weights(net, res, y)
		res = eval_net(net, test)
		for i := 0; i < len(res); i++ {
			fmt.Println(*res[i])
		}
	}
}
