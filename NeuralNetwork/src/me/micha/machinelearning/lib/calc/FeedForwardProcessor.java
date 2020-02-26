package me.micha.machinelearning.lib.calc;

import me.micha.machinelearning.lib.Matrix;
import me.micha.machinelearning.lib.NN;

//Runnable für threadpooled Berechung der Testerfolgsquote
public class FeedForwardProcessor implements Runnable {

	NN dnn;
	Matrix input, answer;
	DoubleObject correct;
	
	public FeedForwardProcessor(NN dnn, Matrix input, Matrix answer, DoubleObject correct) {
		this.dnn = dnn;
		this.input = input;
		this.answer = answer;
		this.correct = correct;
	}
	
	@Override
	public void run() {
		Matrix out = dnn.feedforward(input);
		
		//Hat Index der assoziierten Ziffer die höchste Wahrscheinlichkeit?
		if(answer.maxArgCol(0) == out.maxArgCol(0)) {
			correct.increment();
		}
	}

}
