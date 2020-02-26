package me.micha.machinelearning.lib.learn;

import me.micha.machinelearning.lib.ActivationFunction;
import me.micha.machinelearning.lib.Matrix;

public class AdamPropagation extends ALearnRule {

	public double learningRate, beta1, beta2;
	
	//Moving Averages v, m für die Gewichte und Biases
	Matrix mT, vT, bMT, bVT;
	double iter = 0;
	double epsilon = 0;
		
	public AdamPropagation(ActivationFunction activationFunction, double learningRate, double beta1, double beta2) {
		this(activationFunction, learningRate, beta1, beta2, 1E-8);
	}
	
	public AdamPropagation(ActivationFunction activationFunction, double learningRate, double beta1, double beta2, double epsilon) {
		this.activationFunction = activationFunction;
		this.learningRate = learningRate;
		this.beta1 = beta1;
		this.beta2 = beta2;
		this.epsilon = epsilon;
	}
	
	public AdamPropagation(double learningRate, double beta1, double beta2, double epsilon) {
		this.learningRate = learningRate;
		this.beta1 = beta1;
		this.beta2 = beta2;
		this.epsilon = epsilon;
	}
	
	public AdamPropagation(double learningRate, double beta1, double beta2) {
		this(learningRate, beta1, beta2, 1E-8);
	}
	
	@Override
	public void step() {
		iter++;
	}
	
	//(s. Backpropagation.class)
	public Matrix calcGradient(Matrix in, Matrix error) {
		Matrix g = in.clone();
		g = activationFunction.mapDyFunction(g);
		g.multiply(error);
		
		return g;
	}
	
	//(s. Backpropagation.class)
	public Matrix calcDeltaWeight(Matrix gradient, Matrix out) {
		return calcDeltaWeight(Matrix.matrixProduct(gradient, Matrix.transpose(out)));
	}
	
	//Empfehlung: Formel fuer Adam-Optimizer offen haben
	public Matrix calcDeltaBiasWeight(Matrix gradient) {
		//Falls erster Iterationsschritt=>Initialisierung mit m(0)=0
		if(bMT == null) bMT = new Matrix(gradient.rows, gradient.columns);
		
		//Im Prinzip Momentum (vgl. Momentumpropagation.class)
		bMT.multiply(beta1);
		Matrix g2 = gradient.clone();
		g2.multiply(1-beta1);
		bMT.add(g2);
		
		//Falls erster Iterationsschritt=>Initialisierung mit v(0)=0
		if(bVT == null) bVT = new Matrix(gradient.rows, gradient.columns);
		
		//Quadrierte Moving Average
		bVT.multiply(beta2);
		Matrix g3 = gradient.clone();
		g3.squared();
		g3.multiply(1-beta2);
		bVT.add(g3);
		
		Matrix bMTCorrected = bMT.clone();
		bMTCorrected.divide(1-Math.pow(beta1, iter));
		
		Matrix bVTCorrected = bVT.clone();
		bVTCorrected.divide(1-Math.pow(beta2, iter));
		
		bVTCorrected.sqrt();
		bVTCorrected.add(epsilon);
		
		bMTCorrected.divide(bVTCorrected);
		bMTCorrected.multiply(learningRate);
		
		return bMTCorrected;
	}

	//Empfehlung: Formel fuer Adam-Optimizer offen haben
	public Matrix calcDeltaWeight(Matrix gradient) {
		if(mT == null) mT = new Matrix(gradient.rows, gradient.columns);
		
		mT.multiply(beta1);
		Matrix g2 = gradient.clone();
		g2.multiply(1-beta1);
		mT.add(g2);
		
		if(vT == null) vT = new Matrix(gradient.rows, gradient.columns);
		
		vT.multiply(beta2);
		Matrix g3 = gradient.clone();
		g3.squared();
		g3.multiply(1-beta2);
		vT.add(g3);
		
		Matrix mTCorrected = mT.clone();
		mTCorrected.divide(1-Math.pow(beta1, iter));
		
		Matrix vTCorrected = vT.clone();
		vTCorrected.divide(1-Math.pow(beta2, iter));
		
		vTCorrected.sqrt();
		vTCorrected.add(epsilon);
		
		mTCorrected.divide(vTCorrected);
		mTCorrected.multiply(learningRate);
		
		return mTCorrected;
	}

}
