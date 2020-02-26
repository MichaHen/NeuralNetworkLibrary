package me.micha.machinelearning.lib.learn;

import me.micha.machinelearning.lib.ActivationFunction;
import me.micha.machinelearning.lib.Matrix;

public class Backpropagation extends ALearnRule {

	public double learningRate;
	
	//Festlegung der Lernrate
	public Backpropagation(ActivationFunction activationFunction, double learningRate) {
		this.activationFunction = activationFunction;
		this.learningRate = learningRate;
	}
	
	public Backpropagation(double learningRate) {
		this.learningRate = learningRate;
	}
	
	//Gradient gemaess df(IN)/dw * ERROR = Zeigt in Richtung des steilsten Abstiegs
	public Matrix calcGradient(Matrix in, Matrix error) {
		Matrix g = in.clone();
		g = activationFunction.mapDyFunction(g);
		g.multiply(error);
		
		return g;
	}
	
	//Verrechung des Gradienten für die Deltaweights mit Output der letzten Layer
	public Matrix calcDeltaWeight(Matrix gradient, Matrix out) {
		return calcDeltaWeight(Matrix.matrixProduct(gradient, Matrix.transpose(out)));
	}

	//Gradient = calcGradient(Matrix in, Matrix error)
	public Matrix calcDeltaBiasWeight(Matrix gradient) {
		Matrix g = gradient.clone();
		g.multiply(learningRate);
		
		return g;
	}

	//Gradient != Gradient beim BiasWeight
	@Override
	public Matrix calcDeltaWeight(Matrix gradient) {
		Matrix g = gradient.clone();
		g.multiply(learningRate);
		
		return g;
	}

}
