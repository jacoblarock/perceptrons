package main

import (
	"fmt"
)

func main() {
	net := make([]layer, 2)
	net[0] = make_layer(6, 3, 0.5)
	net[1] = make_layer(3, 1, 0.5)
	fmt.Println(net[0].perceps[0].weights[0])
	net_rand_init(net)
	print_net(net)
	test := make([]float32, 6)
	for i := 0; i < len(test); i++ {
		test[i] = 0.1
	}
	res := eval_net(net, test)
	fmt.Println(res)
}
