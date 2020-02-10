package me.micha.machinelearning.layer;

import me.micha.machinelearning.Matrix;

public abstract class ILayer {

	public Matrix weights;
	public Matrix biases;

	public abstract Matrix calcOutput(Matrix input);
	
	public abstract Matrix calcGradient(Matrix in, Matrix error);
	
	public abstract void calcAndAdapt(Matrix in, Matrix error, Matrix out);
	
	public abstract double getLearningRate();
	
	public abstract int getNodes();
	
	public abstract void changeWeightsBiases(Matrix dWeights, Matrix dBias);
	
}
