package me.micha.machinelearning.lib;

public enum ActivationFunction {

	SIGMOID, TANH, RELU, SOFTMAX;
	
	//Ersetzen der Komponenten der Funktionswerte durch ihren Funktionwert der jeweiligen Funktion
	public Matrix mapFunction(Matrix m) {
		switch (this) {
		case SIGMOID:
			for(int i = 0; i < m.matrix.length; i++) {
				m.matrix[i] = sigmoid(m.matrix[i]);
			}
			return m;
		case TANH:
			for(int i = 0; i < m.matrix.length; i++) {
				m.matrix[i] = tanh(m.matrix[i]);
			}
			return m;
		case RELU:
			for(int i = 0; i < m.matrix.length; i++) {
				m.matrix[i] = relu(m.matrix[i]);
			}
			return m;
		case SOFTMAX:
			//Berechnung der Summe, da sie für die Layer konstant ist
			double total = 0;
			for(int i = 0; i < m.rows; i++) {
				total += Math.exp(m.matrix[m.indexFetch(i, 0)]);
			}
			for(int i = 0; i < m.matrix.length; i++) {
				m.matrix[i] = softmax(m.matrix[i], total);
			}
			return m;
		}
		return m;
		
	}
	
	//Ersetzen der Komponenten der Funktionswerte durch ihren Ableitungswert der jeweiligen Funktion
	public Matrix mapDyFunction(Matrix m) {
		switch (this) {
		case SIGMOID:
			for(int i = 0; i < m.matrix.length; i++) {
				m.matrix[i] = dySigmoid(m.matrix[i]);
			}
			return m;
		case TANH:
			for(int i = 0; i < m.matrix.length; i++) {
				m.matrix[i] = dyTanh(m.matrix[i]);
			}
			return m;
		case RELU:
			for(int i = 0; i < m.matrix.length; i++) {
				m.matrix[i] = dyRelu(m.matrix[i]);
			}
			return m;
		case SOFTMAX:
			for(int i = 0; i < m.matrix.length; i++) {
				m.matrix[i] = dySoftmax(m.matrix[i]);
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
	
	//Ableitung eigentlich df(x)/dx=f(x)(1-(f(x))
	//Die Werte y sind durch Feedforward bereits ein Funktionswert der Funktion=> y=f(x) => y * (1-y)
	public static double dySigmoid(double y) {
		return y * (1 - y);
	}
	
	//TANH
	public static double tanh(double x) {
		return Math.tanh(x);
	}

	//Selbes Prinzip wie bei dySigmoid()
	public static double dyTanh(double y) {
		return 1 - Math.pow(y, 2);
	}

	//RELU
	public static double relu(double x) {
		return Math.max(0, x);
	}
	
	public static double dyRelu(double x) {
		return x > 0  ? 1 : 0;
	}
	
	//SOFTMAX
	public static double softmax(double x, Matrix values) {
		double total = 0;
		for(int i = 0; i < values.rows; i++) {
			total += Math.exp(values.matrix[values.indexFetch(i, 0)]);
		}
	    return Math.exp(x) / total;
	}
	
	//Softmax mit vorgegebener Summe => effizientere Berechnung
	public static double softmax(double x, double total) {
	    return Math.exp(x) / total;
	}
	
	//Ableitung nach einer Komponente des Input-Vektors
	//=> logistische Ableitung. Mit dem Prinzip  von dySigmoid() => x * (1-x)
	public static double dySoftmax(double x) {
	    return x * (1 - x);
	}
}
