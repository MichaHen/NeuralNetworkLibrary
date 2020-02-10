package me.micha.machinelearning;

import me.micha.machinelearning.layer.ILayer;

public class NN {

	ILayer[] layers;
	
	public NN(ILayer... layers) {
		this.layers = layers;
		
		for(int i = 1; i < layers.length; i++) {		
			layers[i].weights = new Matrix(layers[i].getNodes(), layers[i-1].getNodes());			
			layers[i].weights.randomize();
			
			layers[i].biases = new Matrix(layers[i].getNodes(), 1);
			layers[i].biases.randomize();
		}
	}
	
	public Matrix feedforward(Matrix input) {
		Matrix out = input;
		for(int i = 0; i < layers.length; i++) {
			out = layers[i].calcOutput(out);
		}
		
		return out;
	}
	
	public void backpropagate(Matrix input, Matrix answer) {
		// HIDDEN OUTPUTS
		Matrix[] out = new Matrix[layers.length];
		out[0] = input;
		for(int i = 0; i < out.length - 1; i++) {
			out[i+1] = layers[i+1].calcOutput(out[i]);
		}
		
		Matrix[] errors = new Matrix[out.length];
		errors[errors.length - 1] = answer.clone();
		errors[errors.length - 1].subtract(out[out.length - 1]);
		
		for(int i = layers.length - 2; i > -1; i--) {
			Matrix g = layers[i+1].calcGradient(out[i+1], errors[i+1]);
			
			this.layers[i+1].weights.add(Matrix.matrixProduct(g,  Matrix.transpose(out[i])));
			this.layers[i+1].biases.add(g);
			
			errors[i] = Matrix.matrixProduct(Matrix.transpose(layers[i+1].weights), errors[i+1]);
		}
	}
	
	public void backpropagate(Matrix[] out, Matrix error) {
		Matrix[] errors = new Matrix[out.length];
		errors[errors.length - 1] = error;
		
		for(int i = layers.length - 2; i > -1; i--) {
			layers[i+1].calcAndAdapt(out[i+1], errors[i+1], out[i]);
			errors[i] = Matrix.matrixProduct(Matrix.transpose(layers[i+1].weights), errors[i+1]);
		}
	}
	
	//TRAINING METHODS
	public void train(TrainingData data, int batchSize) {
		batchSize = Math.max(1, batchSize);
		Matrix avgError = new Matrix(data.getAnswerSize(), 1);
		Matrix[] avgOut = new Matrix[layers.length];
		
		for(int di = 0; di < data.length(); di++) {
			Entry ent = data.getEntry(di);
			int d = (di+1) % batchSize;
			
			Matrix[] out = new Matrix[layers.length];
			out[0] = ent.getInput();
			for(int i = 0; i < out.length - 1; i++) {
				out[i+1] = layers[i+1].calcOutput(out[i]);
			}
			
			Matrix error = ent.getAnswer();
			error.subtract(feedforward(ent.getInput()));
			error.divide(batchSize);
			avgError.add(error);
			
			for(int i = 0; i < avgOut.length; i++) {
				Matrix o = out[i];
				o.divide(batchSize);
				if(avgOut[i] == null) avgOut[i] = o;
				avgOut[i].add(o);
			}
			
			if(d == 0) {
				backpropagate(avgOut, avgError);
				avgError.init(0);
				for(Matrix o : avgOut) {
					o.init(0);
				}
			}
		}
	}
	
	public void train(TrainingData data, int epochs, int batchSize) {
		for(int e = 0; e < epochs; e++) {
			train(data, batchSize);
		}
	}
	
}
