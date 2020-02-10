package me.micha.machinelearning.layer;

import me.micha.machinelearning.ActivationFunctions;
import me.micha.machinelearning.Matrix;

public class Layer extends ILayer {

	ActivationFunctions activationFunction;
	double learningRate;
	int nodes;
	
	double momentum;
	Matrix pDeltaWeight, pDeltaBiasWeight;
	
	public Layer(int nodes, ActivationFunctions activationFunction, double learningRate) {
		this(nodes, activationFunction, learningRate, 0);
	}
	
	public Layer(int nodes, ActivationFunctions activationFunction, double learningRate, double momentum) {
		this.nodes = nodes;
		this.activationFunction = activationFunction;
		this.learningRate = learningRate;
		this.momentum = momentum;
	}
	
	public Matrix calcOutput(Matrix input) {
		Matrix out = Matrix.matrixProduct(weights, input);
		out.add(biases);
		out = activationFunction.mapFunction(out);
		return out;
	}
	
	public Matrix calcGradient(Matrix in, Matrix error) {
		Matrix g = in.clone();
		g = activationFunction.mapDyFunction(g);
		g.multiply(error);
		g.multiply(this.learningRate);
		
		return g;
	}
	
	@Override
	public void calcAndAdapt(Matrix in, Matrix error, Matrix out) {
		Matrix g = calcGradient(in, error);
		
		changeWeightsBiases(Matrix.matrixProduct(g, Matrix.transpose(out)), g);
	}
	
	public void changeWeightsBiases(Matrix dWeights, Matrix dBias) {
		if(momentum != 0) {
			if(pDeltaBiasWeight != null && pDeltaWeight != null) {
				dWeights.multiply(1-momentum);
				pDeltaWeight.multiply(momentum);
				dWeights.add(pDeltaWeight);
				
				dBias.multiply(1-momentum);
				pDeltaBiasWeight.multiply(momentum);
				dBias.add(pDeltaBiasWeight);
			}
			
			pDeltaWeight = dWeights;
			pDeltaBiasWeight = dBias;
		}
		
		weights.add(dWeights);
		biases.add(dBias);
	}
	
	@Override
	public double getLearningRate() {
		return learningRate;
	}
	
	public int getNodes() {
		return nodes;
	}
	
	public Matrix getPreviousDeltaWeight() {
		return pDeltaWeight;
	}
	
	public Matrix getPreviousDeltaBiasWeight() {
		return pDeltaBiasWeight;
	}
	
	public double getMomentum() {
		return momentum;
	}

}
