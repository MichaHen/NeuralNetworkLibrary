package me.micha.machinelearning.lib.layer;

import me.micha.machinelearning.lib.ActivationFunction;
import me.micha.machinelearning.lib.Matrix;
import me.micha.machinelearning.lib.learn.ALearnRule;
import me.micha.machinelearning.lib.learn.Backpropagation;

//Vollvermaschte Layer (Feedforward-Layer)
public class DenseLayer extends ALayer {

	int nodes;
	ActivationFunction activationFunction;
	
	public DenseLayer(int nodes, ActivationFunction activationFunction, double learningRate) {
		this(nodes, activationFunction, new Backpropagation(activationFunction, learningRate));
	}
	
	public DenseLayer(int nodes, ActivationFunction activationFunction, ALearnRule rule) {
		this.nodes = nodes;
		this.activationFunction = activationFunction;
		this.rule = rule;
		this.rule.activationFunction = activationFunction;
	}
	
	//Feedforward gemaess: f(O[l-1] * W[l] + B[l])
	@Override
	public Matrix calcOutput(Matrix input) {
		Matrix out = Matrix.matrixProduct(weights, input);
		out.add(biases);
		out = activationFunction.mapFunction(out);
		
		return out;
	}

	//Weiterleitung zur jeweiligen Lernregel
	@Override
	public Matrix calcGradient(Matrix in, Matrix error) {
		return rule.calcGradient(in, error);
	}
	
	//Berechung der Deltaweights und DIREKTE Anpassung dieser!
	@Override
	public void calcAndAdapt(Matrix in, Matrix error, Matrix out) {
		Matrix g = calcGradient(in, error);
		changeWeightsBiases(rule.calcDeltaWeight(g, out), rule.calcDeltaBiasWeight(g));
	}
	
	@Override
	public ActivationFunction getActivationFunction() {
		return activationFunction;
	}

	@Override
	public int getNodes() {
		return nodes;
	}

	@Override
	public void changeWeightsBiases(Matrix dWeights, Matrix dBias) {
		weights.add(dWeights);
		biases.add(dBias);
	}


}
