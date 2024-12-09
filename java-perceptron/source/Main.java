import java.util.Arrays;
import java.util.Random;

public class Main {
	public static void main(String[] args) throws Exception {
		Random rand = new Random();
		int size = 25000000;
		float[][] data = new float[size][3];
		int[] expects = new int[size];
		for (int i = 0; i < size; i++) {
			int x = rand.nextInt(100) - 50;
			int y = rand.nextInt(100) - 50;
			data[i][0] = 1;
			data[i][1] = x;
			data[i][2] = y;
			if (y > 0) {
				expects[i] = 1;
			} else {
				expects[i] = -1;
			}
		}
		Perceptron p = new Perceptron(3, 0.2f);
		System.out.println(p.evaluate(data, expects));
		System.out.println(Arrays.toString(p.weights));
		p.bulkTrain(data, expects);
		System.out.println(p.evaluate(data, expects));
		System.out.println(Arrays.toString(p.weights));
		p.bulkTrain(data, expects);
		System.out.println(p.evaluate(data, expects));
		System.out.println(Arrays.toString(p.weights));
		p.bulkTrain(data, expects);
		System.out.println(p.evaluate(data, expects));
		System.out.println(Arrays.toString(p.weights));
	}
}
