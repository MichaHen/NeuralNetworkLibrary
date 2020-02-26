package me.micha.machinelearning.lib.layer;

import me.micha.machinelearning.lib.ActivationFunction;
import me.micha.machinelearning.lib.Matrix;
import me.micha.machinelearning.lib.learn.ALearnRule;

public abstract class ALayer {
	//Gewichtsmatrix, die von der vorherigen Layer zu dieser "zeigt"
	public Matrix weights;
	//Biasmatrix dieser Layer
	public Matrix biases;
	//Lernregel. Momentum/Adam/SGD/...
	public ALearnRule rule;
	
	//Feedforward-Methode. Output der letzten Layer als Input.
	public abstract Matrix calcOutput(Matrix input);
	
	//Berechnung des Gradienten
	public abstract Matrix calcGradient(Matrix in, Matrix error);
	
	//Input von Output der letzten Layer. Output dieser Layer und dem Fehler dieser Layer. Anpassung der Gewichte und Biases.
	public abstract void calcAndAdapt(Matrix in, Matrix error, Matrix out);
	
	public abstract ActivationFunction getActivationFunction();
	
	public abstract int getNodes();
	
	//Folgemethode von calcAndAdapt()
	public abstract void changeWeightsBiases(Matrix dWeights, Matrix dBias);
	
}
