package me.micha.machinelearning.lib.trainingdata;

import me.micha.machinelearning.lib.Matrix;

public class Entry {

	Matrix input, answer;
	
	//Datenpaar-Objekt. Für MNIST Bilddaten und One-Hot-Encoded-Vektor für die jeweilige Ziffer
	public Entry(Matrix input, Matrix answer) {
		this.input = input;
		this.answer = answer;
	}
	
	public Matrix getInput() {
		return input;
	}
	
	public Matrix getAnswer() {
		return answer;
	}
	
}
