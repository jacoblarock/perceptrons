#include <stdlib.h>
#include <stdio.h>
#include <limits.h> 
#include <stdint.h> 

const int size = 3;

typedef struct perceptron {
	double weights[3];
	double learn_rate;
} perceptron;

int classify(perceptron* perceptron, double values[size]) {
	double sum = 0;
	for (int i = 0; i < size; i++)
		sum += perceptron->weights[i] * values[i];
	if (sum > 0)
		return 1;
	return -1;
}

void train(perceptron* perceptron, double values[size], int expect) {
	if (classify(perceptron, values) != expect)
		for (int i = 0; i < size; i++) {
			perceptron->weights[i] = perceptron->weights[i] + 
				perceptron->learn_rate * (double) expect * values[i];
		}
}

void bulk_train(perceptron* perceptron, double value_list[][size], int expects[]) {
	for (int i = 0; i < 100000; i++)
		train(perceptron, value_list[i], expects[i]);
}

int evaluate(perceptron* perceptron, double value_list[][size], int expects[]) {
	int wins = 0;
	for (int i = 0; i < 100000; i++)
		if (classify(perceptron, value_list[i]) == expects[i])
			wins++;
	return wins;
}

int main() {
	perceptron* p = (perceptron*) malloc(sizeof(perceptron));
	p->learn_rate = 0.2;
	for (int i = 0; i < size; i++)
		p->weights[i] = 1;
	for (int round = 0; round < 100; round++) {
	int data_size = 250000;
	double data[data_size][size];
	int expects[data_size];
		for (int i = 0; i < data_size; i++) {
			int x = rand() % 100 - 50;
			int y = rand() % 100 - 50;
			data[i][0] = 1;
			data[i][1] = x;
			data[i][2] = y;
			if (y > 0)
				expects[i] = 1;
			else
				expects[i] = -1;
		}
		// printf("\n%i\n", evaluate(p, data, expects));
		// for (int i = 0; i < size; i++)
			// printf("%f ", p->weights[i]);
		bulk_train(p, data, expects);
		// printf("\n%i\n", evaluate(p, data, expects));
		// for (int i = 0; i < size; i++)
			// printf("%f ", p->weights[i]);
		bulk_train(p, data, expects);
		// printf("\n%i\n", evaluate(p, data, expects));
		// for (int i = 0; i < size; i++)
			// printf("%f ", p->weights[i]);
		bulk_train(p, data, expects);
	}
}
