public class Perceptron {
	int size;
	float learnRate;
	float[] weights;

	public Perceptron(int size, float learnRate) {
		this.size = size;
		this.learnRate = learnRate;
		this.weights = new float[size];
		for (int i = 0; i < size; i++) {
			this.weights[i] = 1;
		}
	}

	public int classify(float[] values) throws Exception{
		if (values.length != weights.length)
			throw new Exception("incorrect length of values");
		float sum = 0;
		for (int i = 0; i < weights.length; i++)
			sum += weights[i] * values[i];
		if (sum > 0)
			return 1;
		return -1;
	}

	public void train(float[] values, int expect) throws Exception{
		if (classify(values) != expect) {
			for (int i = 0; i < weights.length; i++)
				weights[i] = weights[i] + learnRate * expect * values[i];
		}
	}

	public void bulkTrain(float[][] valueList, int[] expects) throws Exception{
		if (valueList.length != expects.length)
			throw new Exception("the lengths of valueList and expects must correspond");
		for (int i = 0; i < valueList.length; i++)
			train(valueList[i], expects[i]);
	}

	public int evaluate(float[][] valueList, int[] expects) throws Exception{
		if (valueList.length != expects.length)
			throw new Exception("the lengths of valueList and expects must correspond");
		int wins = 0;
		for (int i = 0; i < valueList.length; i++) {
			if (classify(valueList[i]) == expects[i])
				wins++;
		}
		return wins;
	}
}
