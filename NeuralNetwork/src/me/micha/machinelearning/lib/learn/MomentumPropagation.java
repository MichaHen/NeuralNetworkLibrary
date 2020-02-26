package me.micha.machinelearning.lib.learn;

import me.micha.machinelearning.lib.ActivationFunction;
import me.micha.machinelearning.lib.Matrix;

public class MomentumPropagation extends ALearnRule {

	public double learningRate, momentum;
	
	//Gewichtsänderung des letzten Iterationsschritts
	Matrix pDeltaWeight, pDeltaBiasWeight;
	
	//Festlegung Momentum gamma und Lernrate
	public MomentumPropagation(ActivationFunction activationFunction, double learningRate, double momentum) {
		this.activationFunction = activationFunction;
		this.learningRate = learningRate;
		this.momentum = momentum;
	}
	
	public MomentumPropagation(double learningRate, double momentum) {
		this.learningRate = learningRate;
		this.momentum = momentum;
	}
	
	// (s. Backpropagation.class)
	public Matrix calcGradient(Matrix in, Matrix error) {
		Matrix g = in.clone();
		g = activationFunction.mapDyFunction(g);
		g.multiply(error);
		
		return g;
	}
	
	// (s. Backpropagation.class)
	public Matrix calcDeltaWeight(Matrix gradient, Matrix out) {
		return calcDeltaWeight(Matrix.matrixProduct(gradient, Matrix.transpose(out)));
	}
	
	public Matrix calcDeltaBiasWeight(Matrix gradient) {
		Matrix dBias = gradient.clone();
		//Falls erster Iterationsschritt=>Initliasierung, Annahme von W(0)=0
		if(pDeltaBiasWeight == null) {
			pDeltaBiasWeight = new Matrix(dBias.rows, dBias.columns, 0);
		}
		
		//Gewichtung durch Momentum
		dBias.multiply(1-momentum);
		pDeltaBiasWeight.multiply(momentum);
		dBias.add(pDeltaBiasWeight);
		
		dBias.multiply(learningRate);
		
		//Update der letzten Gewichtsänderung = Iterationsschritt beendet
		pDeltaBiasWeight = dBias;
		
		return dBias;
	}

	@Override
	public Matrix calcDeltaWeight(Matrix gradient) {
		Matrix dWeight = gradient.clone();
		//Falls erster Iterationsschritt=>Initliasierung, Annahme von W(0)=0
		if(pDeltaWeight == null) {
			pDeltaWeight = new Matrix(dWeight.rows, dWeight.columns, 0);
		}
		
		//Gewichtung durch Momentum
		dWeight.multiply(1-momentum);
		pDeltaWeight.multiply(momentum);
		dWeight.add(pDeltaWeight);
		
		dWeight.multiply(learningRate);
		
		//Update der letzten Gewichtsänderung = Iterationsschritt beendet
		pDeltaWeight = dWeight;
		
		return dWeight;
	}

}
