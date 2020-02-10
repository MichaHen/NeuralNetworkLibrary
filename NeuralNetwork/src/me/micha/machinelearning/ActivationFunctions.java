package me.micha.machinelearning;

public enum ActivationFunctions {

	SIGMOID, TANH, RELU, SOFTMAX;
	
	ActivationFunctions() {
		
	}
	
	public double f(double x) {
		switch (this.ordinal()) {
		case 0:
			return sigmoid(x);
		case 1:
			return tanh(x);
		case 2:
			return relu(x);
		default:
			return 0;
		}
	}
	
	public Matrix mapFunction(Matrix m) {
		switch (this.ordinal()) {
		case 0:
			for(int i = 0; i < m.rows; i++) {
				for(int j = 0; j < m.columns; j++) {
					m.matrix[i][j] = sigmoid(m.matrix[i][j]);
				}
			}
			return m;
		case 1:
			for(int i = 0; i < m.rows; i++) {
				for(int j = 0; j < m.columns; j++) {
					m.matrix[i][j] = tanh(m.matrix[i][j]);
				}
			}
			return m;
		case 2:
			for(int i = 0; i < m.rows; i++) {
				for(int j = 0; j < m.columns; j++) {
					m.matrix[i][j] = relu(m.matrix[i][j]);
				}
			}
			return m;
		case 3:
			double total = 0;
			for(int i = 0; i < m.rows; i++) {
				total += Math.exp(m.matrix[i][0]);
			}
			for(int i = 0; i < m.rows; i++) {
				for(int j = 0; j < m.columns; j++) {
					m.matrix[i][j] = softmax(m.matrix[i][j], total);
				}
			}
			return m;
		}
		return m;
		
	}
	
	public Matrix mapDyFunction(Matrix m) {
		switch (this.ordinal()) {
		case 0:
			for(int i = 0; i < m.rows; i++) {
				for(int j = 0; j < m.columns; j++) {
					m.matrix[i][j] = dySigmoid(m.matrix[i][j]);
				}
			}
			return m;
		case 1:
			for(int i = 0; i < m.rows; i++) {
				for(int j = 0; j < m.columns; j++) {
					m.matrix[i][j] = dyTanh(m.matrix[i][j]);
				}
			}
			return m;
		case 2:
			for(int i = 0; i < m.rows; i++) {
				for(int j = 0; j < m.columns; j++) {
					m.matrix[i][j] = dyRelu(m.matrix[i][j]);
				}
			}
			return m;
		case 3:
			for(int i = 0; i < m.rows; i++) {
				for(int j = 0; j < m.columns; j++) {
					m.matrix[i][j] = dySoftmax(m.matrix[i][j]);
				}
			}
			return m;
		}
		
		return m;
	}
	
	//FUNCTIONS
	
	//SIGMOID
	public static double sigmoid(double x) {
		return 1 / (1 + Math.exp(-x));
	}
	
	public static double dSigmoid(double x) {
		return 1 / (1 + Math.exp(-x));
	}
	
	public static double dySigmoid(double y) {
		return y * (1 - y);
	}
	
	//TANH
	public static double tanh(double x) {
		return Math.tanh(x);
	}
	
	public static double dTanh(double x) {
		return 1 / Math.pow(Math.cosh(x), 2);
	}
	
	public static double dyTanh(double y) {
		return 1 - Math.pow(y, 2);
	}

	//RELU
	public static double relu(double x) {
		return Math.max(0, x);
	}
	
	public static double dRelu(double x) {
		return x > 0  ? 1 : 0;
	}
	
	public static double dyRelu(double x) {
		return x > 0  ? 1 : 0;
	}
	
	//SOFTMAX
	public static double softmax(double x, Matrix values) {
		double total = 0;
		for(int i = 0; i < values.rows; i++) {
			total += Math.exp(values.matrix[i][0]);
		}
	    return Math.exp(x) / total;
	}
	
	public static double softmax(double x, double total) {
	    return Math.exp(x) / total;
	}
	
	public static double dySoftmax(double x) {
	    return x * (1 - x);
	}
}
