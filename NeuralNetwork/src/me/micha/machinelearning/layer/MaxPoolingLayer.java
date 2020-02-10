package me.micha.machinelearning.layer;

import me.micha.machinelearning.Matrix;

public class MaxPoolingLayer {

	public MaxPoolingLayer() {
		
	}
	
	public Matrix calcOutput(Matrix weights, Matrix input, Matrix biases) {
		if(input.rows % 4 != 0 || input.columns % 4 != 0) return input;
		Matrix out = new Matrix(input.rows/4, input.columns/4);
		
		for(int i = 0; i < out.rows; i++) {
			for(int j = 0; j < out.columns; j++) {
				out.matrix[i][j] = max(input.matrix[2*i][2*j], input.matrix[2*i+1][2*j], input.matrix[2*i][2*j+1], input.matrix[2*i+1][2*j+1]);
			}
		}
		
		return out;
	}
	
	private double max(double a, double b, double c, double d) {
		return Math.max(a, Math.max(b, Math.max(c, d)));
	}

}
