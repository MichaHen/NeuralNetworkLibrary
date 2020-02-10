package me.micha.machinelearning;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import me.micha.machinelearning.layer.ILayer;
import me.micha.machinelearning.layer.InputLayer;
import me.micha.machinelearning.layer.Layer;
import me.micha.machinelearning.mnist.MnistConverter;

public class Main {

	public static final int EPOCHS = 100;
	
 	public static void main(String[] args) {
		MnistConverter.load();
		calculate("test");
//		new Thread(new Runnable() {
//			
//			@Override
//			public void run() {
//				calculate("t1");
//			}
//		}).start();
//		new Thread(new Runnable() {
//			
//			@Override
//			public void run() {
//				calculate("t2");
//			}
//		}).start();
//		new Thread(new Runnable() {
//					
//					@Override
//					public void run() {
//						calculate("t3");
//					}
//				}).start();
//		new Thread(new Runnable() {
//			
//			@Override
//			public void run() {
//				calculate("t4");
//			}
//		}).start();
	}
 	
 	public static void calculate(String dataAppendix) {
 		double[] rates = new double[EPOCHS];
 		TrainingData data = MnistConverter.getData();
		TrainingData testData = MnistConverter.getTestData();
		
		NN dnn = new NN(new InputLayer(784),
				new Layer(256, ActivationFunctions.SIGMOID, 0.01),
				new Layer(10, ActivationFunctions.SIGMOID, 0.001));
		
		System.out.println("TRAININGSET: " + data.length() + " entries");
		System.out.println("TESTSET: " + testData.length() + " entries");
		System.out.println("STARTING TRAINING...");
		
		System.out.println(dnn.feedforward(data.getEntry(0).getInput()));
		
		for(int e = 0; e < EPOCHS; e++) {
			long start = System.currentTimeMillis();
			System.out.println("Epoche: " + e);
			dnn.train(data, 1);
			double r = guessAndCalcRate(dnn, testData);
			System.out.println(r);
			rates[e] = r;
			System.out.println("TIME: " + ((System.currentTimeMillis() - start)/1000) + "s");
//			dnn.learningRate -= dnn.learningRateDecay;
		}
		
		System.out.println("Finished: Saving data");
		
		String path = "data_" + dataAppendix + ".txt";
 		File file = new File(path);
		
		try {
			file.createNewFile();
			BufferedWriter writer = new BufferedWriter(new FileWriter(path, true));
			writer.append(getNetworkInfo(dnn));
			
			for(int i = 0; i < EPOCHS; i++) {
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
 	
 	private static String getNetworkInfo(NN nn) {
 		StringBuilder b = new StringBuilder("NN-");
 		for(ILayer l : nn.layers) {
 			b.append(l.getNodes() + "-");
 		}
 		b.setLength(b.length()-1);
 		b.append(System.lineSeparator());
 		for(int i = 0; i < nn.layers.length; i++) {
 			ILayer l = nn.layers[i];
 			b.append("Layer " + i + ": " + l.getLearningRate() + ", ");
 		}
 		b.append(System.lineSeparator());
 		
 		return b.toString();
 	}
 	
 	public static double guessAndCalcRate(NN dnn, TrainingData testData) {
 		double total = 0;
		double correct = 0;
		
		for(int i = 0; i < testData.length(); i++) {
			Entry e = testData.getEntry(i);
			int guess = dnn.feedforward(e.getInput()).maxArgCol(0);
			
			if(guess == e.getAnswer().maxArgCol(0)) {
				correct++;
			}
			
			total++;
		}
		
		return (correct / total);
 	}
 	
//	public static void printArray(double[] array) {
//		for(double d : array) {
//			System.out.println(d);
//		}
//	}
	
}
