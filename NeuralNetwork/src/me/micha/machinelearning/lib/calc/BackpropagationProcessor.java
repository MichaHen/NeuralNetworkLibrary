package me.micha.machinelearning.lib.calc;

import me.micha.machinelearning.lib.Matrix;
import me.micha.machinelearning.lib.NN;
import me.micha.machinelearning.lib.trainingdata.DataSet;
import me.micha.machinelearning.lib.trainingdata.Entry;
//Backpropagation eines Datenpaares
public class BackpropagationProcessor implements Runnable {

	public Matrix[] dWGrad, dBGrad;
	DataSet data;
	NN dnn;
	int batchSize;
	
	public BackpropagationProcessor(NN dnn, DataSet data, int batchSize, Matrix[] dWGrad, Matrix[] dBGrad) {
		this.dnn = dnn;
		this.batchSize = batchSize;
		this.data = data;
		this.dWGrad = dWGrad;
		this.dBGrad = dBGrad;
	}
	
	@Override
	public void run() {
		//Zufällige Auswahl eines Datenpaares. Ohne dies ist Batch-Training deutlich ineffizienter und wirkungsloser, da immer dieselben Daten zusammengefasst werden
		Entry ent = data.randomData();
		
		//Feedwardward-Teil
		Matrix[] out = dnn.feedforwardMem(ent.getInput());
		
		//Fehlerberechung
		Matrix error = dnn.lossFunction.dE(out[out.length-1], ent.getAnswer());
		
		Matrix[] errors = new Matrix[out.length];
		errors[errors.length - 1] = error;
		
		//Fehlerrückführung
		for(int i = dnn.layers.length - 2; i > -1; i--) {
			errors[i] = Matrix.matrixProduct(Matrix.transpose(dnn.layers[i+1].weights), errors[i+1]);
			Matrix g = dnn.layers[i+1].rule.calcGradient(out[i+1], errors[i+1]);
			Matrix dW = Matrix.matrixProduct(g, Matrix.transpose(out[i]));
			//Mittelung im Batch. Anteil vorher berechnet=> Verhindert Zwischenspeichern mehrerer Daten
			g.divide((double)batchSize);
			dW.divide((double)batchSize);
			
			//Wenn Backpropagation des ersten Datenpaares=> Initialisierung der Matrix
			if(dWGrad[i+1] == null) dWGrad[i+1] = new Matrix(dW.rows, dW.columns);
			if(dBGrad[i+1] == null) dBGrad[i+1] = new Matrix(g.rows, g.columns);
			
			//Summierung der Gradienten
			dWGrad[i+1].add(dW);
			dBGrad[i+1].add(g);
		}
	}
	
}
