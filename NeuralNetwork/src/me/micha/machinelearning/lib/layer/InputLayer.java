package me.micha.machinelearning.lib.layer;

import me.micha.machinelearning.lib.ActivationFunction;
import me.micha.machinelearning.lib.Matrix;

//Bildet Inputs ab
public class InputLayer extends ALayer {

	int nodes;
	
	//Keine Aktivierungsfunktion oder Lernrate
	public InputLayer(int nodes) {
		this.nodes = nodes;
	}
	
	//Kopieren der GesamtInputDaten des Netzwerks => Keine Veränderung der Trainings- oder Testdaten
	@Override
	public Matrix calcOutput(Matrix input) {
		return input.clone();
	}
	
	//Keine Anpassung von Gewichten => Kein Lernen.
	@Override
	public Matrix calcGradient(Matrix in, Matrix error) {
		return null;
	}

	@Override
	public int getNodes() {
		return nodes;
	}

	@Override
	public void changeWeightsBiases(Matrix dWeights, Matrix dBias) {}

	@Override
	public void calcAndAdapt(Matrix in, Matrix error, Matrix out) {}

	@Override
	public ActivationFunction getActivationFunction() {
		return null;
	}

}
