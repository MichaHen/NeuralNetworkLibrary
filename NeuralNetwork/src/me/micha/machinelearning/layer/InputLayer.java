package me.micha.machinelearning.layer;

import me.micha.machinelearning.Matrix;

public class InputLayer extends ILayer {

	int nodes;
	
	public InputLayer(int nodes) {
		this.nodes = nodes;
	}
		
	@Override
	public Matrix calcOutput(Matrix input) {
		return input;
	}

	@Override
	public Matrix calcGradient(Matrix in, Matrix error) {
		return null;
	}

	@Override
	public double getLearningRate() {
		return 0;
	}

	@Override
	public int getNodes() {
		return nodes;
	}

	@Override
	public void changeWeightsBiases(Matrix dWeights, Matrix dBias) {}

	@Override
	public void calcAndAdapt(Matrix in, Matrix error, Matrix out) {}

}
