package me.micha.machinelearning.lib;

public enum LossFunction {

	MSE, CROSS_ENTROPY;
	
	//Berechnung der Ableitung der Fehlerfunktion=Berechung des Errors. Die eigentliche Funktion ist nicht relevant, da sie nur zur Backpropagation genutzt wird.
	public double dE(double o, double t) {
		switch (this) {
			case MSE:
					return (t-o);
			case CROSS_ENTROPY:
					return (-t/o);
		}
		return 0;
	}
	
	//Berechnung des Errors für die Gesamte Output-Layer: (ZIEL - OUTPUT)
	public Matrix dE(Matrix o, Matrix t) {
		Matrix m = new Matrix(o.rows, o.columns);
		
		for(int i = 0; i < o.matrix.length; i++) {
			m.matrix[i] = dE(o.matrix[i], t.matrix[i]);
		}
		
		return m;
	}
	
}
