package me.micha.machinelearning.mnist;

import java.util.List;

import me.micha.machinelearning.Matrix;
import me.micha.machinelearning.TrainingData;

public class MnistConverter {

	private static TrainingData data, testData;
	
	public static void load() {
		System.out.println("Loading data...");
		int[] labels = MnistReader.getLabels("train-labels.idx1-ubyte");
		List<int[][]> images = MnistReader.getImages("train-images.idx3-ubyte");
		
		int[] labels2 = MnistReader.getLabels("t10k-labels.idx1-ubyte");
		List<int[][]> images2 = MnistReader.getImages("t10k-images.idx3-ubyte");
		
		data = new TrainingData(labels.length);
		
		for(int i = 0; i < labels.length; i++) {
			data.addEntry(Matrix.fromArray(toInput(images.get(i))), Matrix.fromArray(toAnswer(labels[i])));
		}
		
		testData = new TrainingData(labels2.length);
		
		for(int i = 0; i < labels2.length; i++) {
			testData.addEntry(Matrix.fromArray(toInput(images2.get(i))), Matrix.fromArray(toAnswer(labels2[i])));
		}
		System.out.println("Loaded data!");
	}
	
	public static TrainingData getData() {
		return data;
	}
	
	public static TrainingData getTestData() {
		return testData;
	}
	
	private static double[] toInput(int[][] matrix) {
 		double[] array = new double[784];
 		
 		int index = 0;
 		
 		for(int i = 0; i < 27; i++) {
 			for(int j = 0; j < 27; j++) {
 				array[index] = matrix[i][j];
 				index++;
 			}
 		}
 		
 		return array;
 	}
 	
 	private static double[] toAnswer(int label) {
 		double[] array = new double[] {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
 		array[label] = 1;
 		
 		return array;
 		
 	}
	
}
