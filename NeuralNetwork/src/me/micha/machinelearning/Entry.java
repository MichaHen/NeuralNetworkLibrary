package me.micha.machinelearning;

public class Entry {

	Matrix input, answer;
	
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
