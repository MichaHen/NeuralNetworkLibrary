package me.micha.machinelearning.test;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import me.micha.machinelearning.lib.ActivationFunction;
import me.micha.machinelearning.lib.NN;
import me.micha.machinelearning.lib.calc.NNExecutor;
import me.micha.machinelearning.lib.layer.DenseLayer;
import me.micha.machinelearning.lib.layer.InputLayer;
import me.micha.machinelearning.lib.learn.AdamPropagation;
import me.micha.machinelearning.lib.learn.MomentumPropagation;
import me.micha.machinelearning.lib.mnist.MnistConverter;

public class Experiment {

	public static final int THREADS = 7;
	
	public static final int POINTS = 100;
	
	public static void main(String[] args) {
		MnistConverter.load(args.length > 0 ? args[0] : "");
		
		testStandardModel();
		
		experimentLearningRate();
		
		experimentBatchSize();
		
		experimentStructure();
		
		experimentMomentum();
		
		experimentAdam();
		
		experimentFinal();
	}
	
	public static void testStandardModel() {
		NN dnn = new NN(new InputLayer(784),
						new DenseLayer(32, ActivationFunction.SIGMOID, 0.01),
						new DenseLayer(16, ActivationFunction.RELU, 0.01),
						new DenseLayer(10, ActivationFunction.SOFTMAX, 0.01));
		
		dnn.trainThreadPooled(MnistConverter.getData(), 10, THREADS);
		
		dnn.test(MnistConverter.getTestData(), THREADS);
	}
	
	//Variation der Lernrate
	public static void experimentLearningRate() {
		double learningRateFrom = 0.5;
		double learningRateTo = 0.00001;
		
		//Startlernrate mit Addition, da am Anfang der Schleife subtrahiert wird
		double learningRate = learningRateFrom + (learningRateFrom-learningRateTo)/(double)POINTS;
		
		//Anpassung in POINTS gleichen Abstaenden
		for(int i = 0; i < POINTS; i++) {
			learningRate -= (learningRateFrom-learningRateTo)/(double)POINTS;
			
			NN dnn = new NN(new InputLayer(784),
					//Veraenderung der Lernrate
					new DenseLayer(32, ActivationFunction.SIGMOID, learningRate),
					new DenseLayer(16, ActivationFunction.RELU, learningRate),
					new DenseLayer(10, ActivationFunction.SOFTMAX, learningRate));
			
			
			double[] accuracy = trainNN(dnn, 10, 50, 4);
			saveData(accuracy, "LR");
		}
	}
	
	//Variation der Batchsize
	public static void experimentBatchSize() {
		//Primfaktoren von 60000 bis 100
		int[] batchSizes = new int[] {1,2,3,4,5,6,8,10,12,15,16,20,24,25,30,32,40,48,50,60,70,75,80,96,100};
		
		//Anpassung in POINTS gleichen Abstaenden
		for(int i = 0; i < batchSizes.length; i++) {
			
			NN dnn = new NN(new InputLayer(784),
					new DenseLayer(32, ActivationFunction.SIGMOID, 0.01),
					new DenseLayer(16, ActivationFunction.RELU, 0.01),
					new DenseLayer(10, ActivationFunction.SOFTMAX, 0.01));
			
			//Veraenderung der Batch-Groesse
			double[] accuracy = trainNN(dnn, batchSizes[i], 50, 4);
			saveData(accuracy, "BATCH");
		}
	}
	
	//Variation der Hidden-Units
	public static void experimentStructure() {
		int firstHidden = 2;
		int secondHidden = 1;
		
		//Hier POINTS=10
		for(int i = 0; i < POINTS; i++) {
			firstHidden *= 2;
			secondHidden *= 2;
			
			NN dnn = new NN(new InputLayer(784),
					//Veraendert die Neuronen in den Hidden-Layers
					new DenseLayer(firstHidden, ActivationFunction.SIGMOID, 0.01),
					new DenseLayer(secondHidden, ActivationFunction.RELU, 0.01),
					new DenseLayer(10, ActivationFunction.SOFTMAX, 0.01));
			
			
			double[] accuracy = trainNN(dnn, 10, 50, 4);
			saveData(accuracy, "STRUCT");
		}
	}
	
	//Variation des Momentums / Analog zur Lernrate
	public static void experimentMomentum() {
		double momentumFrom = 0.95;
		double momentumTo = 0.001;
		
		//Startmomentum mit Addition, da am Anfang der Schleife subtrahiert wird
		double momentum = momentumFrom + (momentumFrom-momentumTo)/(double)POINTS;
		
		//Anpassung in POINTS gleichen Abstaenden
		for(int i = 0; i < POINTS; i++) {
			momentum -= (momentumFrom-momentumTo)/(double)POINTS;
			
			NN dnn = new NN(new InputLayer(784),
					//Veraenderung des Momentums
					new DenseLayer(32, ActivationFunction.SIGMOID, new MomentumPropagation(0.01, momentum)),
					new DenseLayer(16, ActivationFunction.RELU, new MomentumPropagation(0.01, momentum)),
					new DenseLayer(10, ActivationFunction.SOFTMAX, new MomentumPropagation(0.01, momentum)));
			
			
			double[] accuracy = trainNN(dnn, 10, 50, 4);
			saveData(accuracy, "MOM");
		}
	}
	
	//Variation Adam
	public static void experimentAdam() {
		double learningRateFrom = 0.05;
		double learningRateTo = 0.00005;
		
		//Startlernrate mit Addition, da am Anfang der Schleife subtrahiert wird
		double learningRate = learningRateFrom + (learningRateFrom-learningRateTo)/(double)POINTS;
		
		//Anpassung in POINTS gleichen Abstaenden
		for(int i = 0; i < POINTS; i++) {
			learningRate -= (learningRateFrom-learningRateTo)/(double)POINTS;
			
			NN dnn = new NN(new InputLayer(784),
					//Veraenderung der Lernrate
					new DenseLayer(32, ActivationFunction.SIGMOID, new AdamPropagation(learningRate, 0.9, 0.999)),
					new DenseLayer(16, ActivationFunction.RELU, new AdamPropagation(learningRate, 0.9, 0.999)),
					new DenseLayer(10, ActivationFunction.SOFTMAX, new AdamPropagation(learningRate, 0.9, 0.999)));
			
			
			double[] accuracy = trainNN(dnn, 10, 50, 4);
			saveData(accuracy, "ADAM");
		}
	}
	
	//Finale Version
	public static void experimentFinal() {
		//Finalmodell wie beschrieben in Kapitel 6
		NN dnn = new NN(new InputLayer(784),
				new DenseLayer(128, ActivationFunction.SIGMOID, new AdamPropagation(0.00154, 0.9, 0.999)),
				new DenseLayer(64, ActivationFunction.RELU, new AdamPropagation(0.00154, 0.9, 0.999)),
				new DenseLayer(10, ActivationFunction.SOFTMAX, new AdamPropagation(0.00154, 0.9, 0.999)));
		
		
		double[] accuracy = trainNN(dnn, 10, 50, 4);
		saveData(accuracy, "FINAL");
	}
	
	private static double[] trainNN(NN dnn, int batchSize, int epochs, int examples) {
 			NNExecutor[] threads = new NNExecutor[(int) examples];
 			for(int t = 0; t < examples; t++) {
 				threads[t] = new NNExecutor(dnn, epochs, batchSize, THREADS);
 				threads[t].start();
 			}
 			
 			for(NNExecutor n : threads) {
 				try {
					n.join();
				} catch (InterruptedException e) {
					System.out.print("Thread join error: ");
					e.printStackTrace();
				}
 			}
 			
 			double[] avgRates = new double[batchSize+1];
 			
 			for(NNExecutor n : threads) {
 				for(int i = 0; i < n.getRates().length; i++) {
 					avgRates[i] += (n.getRates()[i] / (double)examples);
 				}
 			}
 			
 			return avgRates;
	}
	
	private static void saveData(double[] rates, String appendix) {
 		System.out.println("Saving data...");
		
		String path = "data_" + appendix + ".txt";
 		File file = new File(path);
		
		try {
			file.createNewFile();
			BufferedWriter writer = new BufferedWriter(new FileWriter(path, true));
			
			for(int i = 0; i < rates.length; i++) {
				writer.append((i) + ";" + rates[i]);
				writer.newLine();
			}

			writer.flush();
			writer.close();
			
		} catch (IOException e) {
			e.printStackTrace();
		}
	
		System.out.println("Finished!");
 	}
	
}
