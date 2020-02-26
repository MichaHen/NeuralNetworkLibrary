package me.micha.machinelearning.lib.learn;

import me.micha.machinelearning.lib.ActivationFunction;
import me.micha.machinelearning.lib.Matrix;

public abstract class ALearnRule {

	//Jede (hier implementierte) Lernregel ist abhängig von einer Aktivierungsfunktion
	public ActivationFunction activationFunction;
	
	//Deltaweights von Gradient und Output der letzten Layer
	public abstract Matrix calcDeltaWeight(Matrix gradient, Matrix out);
	
	//gradient = calcDeltaWeigth(Matrix gradient, Matrix out)
	public abstract Matrix calcDeltaWeight(Matrix gradient);
	
	public abstract Matrix calcGradient(Matrix in, Matrix error);
	
	//Bias Deltaweights unabhaengig von einem Ouput, da dieser 1 ist.
	public abstract Matrix calcDeltaBiasWeight(Matrix gradient);

	//Iterationsschritterhoehung
	public void step() {}
	
}
