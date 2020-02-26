package me.micha.machinelearning.lib;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import me.micha.machinelearning.lib.calc.BackpropagationProcessor;
import me.micha.machinelearning.lib.calc.DoubleObject;
import me.micha.machinelearning.lib.calc.FeedForwardProcessor;
import me.micha.machinelearning.lib.layer.ALayer;
import me.micha.machinelearning.lib.trainingdata.DataSet;
import me.micha.machinelearning.lib.trainingdata.Entry;

public class NN {

	public LossFunction lossFunction;
	public ALayer[] layers;
	
	public NN(ALayer... layers) {
		//Standardmäßige Wahl der quad. Fehlerfunktion
		this(LossFunction.MSE, layers);
	}
	
	public NN(LossFunction lossFunction, ALayer... layers) {
		this.lossFunction = lossFunction;
		this.layers = layers;
		
		//Zufällige Initialisierung [-1,1] der Gewicht- und Bias Matrizen. 
		for(int i = 1; i < layers.length; i++) {	
			//Dimension: Nodes der vorherigen Layer x momentane Layer
			layers[i].weights = new Matrix(layers[i].getNodes(), layers[i-1].getNodes());			
			layers[i].weights.randomize();
				
			layers[i].biases = new Matrix(layers[i].getNodes(), 1);
			layers[i].biases.randomize();
		}
	}
	
	//Feedforward gemaeß des Algorithmus in Kapitel 2.3
	//Die Berechnung ist auf das ALayer Objekt ausgelagert
	public Matrix feedforward(Matrix input) {
		Matrix out = input;
		//Durchreichen des Outputs durch alle Layers.
		// Start bei i=1, da die 0-te Layer (Input-Layer) keine Berechnungen durchführt (s. InputLayer.class)
		for(int i = 1; i < layers.length; i++) {
			out = layers[i].calcOutput(out);
		}
		
		return out;
	}
	
	//Feedforward mit Zwischenspeichern der Outputs zur Nutzung im Backpropagation-Algorithmus
	public Matrix[] feedforwardMem(Matrix input) {
		Matrix[] out = new Matrix[layers.length];
		out[0] = input.clone();
		for(int i = 0; i < out.length - 1; i++) {
			out[i+1] = layers[i+1].calcOutput(out[i]);
		}
		
		return out;
	}
	
	//Ausrechnen der Präzision auf den Testdaten
	public double test(DataSet testData, int threads) {
		//Hilf-Wrapper-Objekt, damit die double-value kopiert wird.
		DoubleObject correct = new DoubleObject(0);
		//Ausführen in einem Threadpool, was zu enormen Performancevorteilen führt
		ExecutorService executor = Executors.newFixedThreadPool(threads);
		
		//Hinzufügen der Aufgaben des Threadpools
		for(int i = 0; i < testData.length(); i++) {
			Entry ent = testData.getEntry(i);
			executor.submit(new FeedForwardProcessor(this, ent.getInput(), ent.getAnswer(), correct));
		}
		executor.shutdown();
		try {
			executor.awaitTermination(1, TimeUnit.DAYS);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		return (correct.d / (double)testData.length());
	}
	
	//Backpropagation eines einzelnen Datenpaares
	public void backpropagate(Matrix input, Matrix answer) {
		Matrix[] out = feedforwardMem(input);
		
		Matrix error = lossFunction.dE(out[out.length-1], answer);
		
		backpropagate(out, error);
	}
	
	public void backpropagate(Matrix[] out, Matrix error) {
		Matrix[] errors = new Matrix[out.length];
		errors[errors.length - 1] = error;
		
		//Gehe durch die Layers von hinten nach vorne
		for(int i = layers.length - 2; i > -1; i--) {
			//Error-Anteil am Gesamterror durch Gewichtung
			errors[i] = Matrix.matrixProduct(Matrix.transpose(layers[i+1].weights), errors[i+1]);
			//Iterationsschritt wird erhöht. (Nur für AdamPropagation relevant)
			layers[i+1].rule.step();
			layers[i+1].calcAndAdapt(out[i+1], errors[i+1], out[i]);
		}
	}
	
	//Training einer Epoche mit Batchsize in Threads
	public void trainThreadPooled(DataSet data, int batchSize, int threads) {
		batchSize = Math.max(1, batchSize);
		
		int examples = 0;
		
		//Solange nicht einmal die Anzahl aller Daten durchgegangen wurde
		while(examples < data.length()) {
			//Summierung und Mittelung der Gradienten für Weight und Bias
			Matrix[] dWGrad = new Matrix[layers.length];
			Matrix[] dBGrad = new Matrix[layers.length];
			
			//Thread Pool (s. feedforward())
			ExecutorService executor = Executors.newFixedThreadPool(threads);
			
			//Queueing der Aufgaben. batchSize Datenpaare
			for(int b = 0; b < batchSize; b++) {
				executor.submit(new BackpropagationProcessor(this, data, batchSize, dWGrad, dBGrad));
			}
			executor.shutdown();
			try {
				executor.awaitTermination(1, TimeUnit.DAYS);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			
			//Veränderung der Gewichte nach einer Batch
			for(int i = 1; i < dWGrad.length; i++) {
				layers[i].rule.step();
				layers[i].weights.add(layers[i].rule.calcDeltaWeight(dWGrad[i]));
				layers[i].biases.add(layers[i].rule.calcDeltaBiasWeight(dBGrad[i]));
			}
			
			examples += batchSize;
		}
	}
	
}
