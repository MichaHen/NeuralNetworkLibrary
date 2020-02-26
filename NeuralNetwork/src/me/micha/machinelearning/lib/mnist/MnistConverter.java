package me.micha.machinelearning.lib.mnist;

import java.util.List;

import me.micha.machinelearning.lib.Matrix;
import me.micha.machinelearning.lib.trainingdata.DataSet;

public class MnistConverter {
	
	//Speichern des Trainings- und Testdatensatzes
	private static DataSet data, testData;
	
	public static void load(String path) {
		System.out.println("Loading data...");
		int[] labels = MnistReader.getLabels(path + "train-labels.idx1-ubyte");
		List<int[][]> images = MnistReader.getImages(path + "train-images.idx3-ubyte");
		
		int[] labels2 = MnistReader.getLabels(path + "t10k-labels.idx1-ubyte");
		List<int[][]> images2 = MnistReader.getImages(path + "t10k-images.idx3-ubyte");
		
		data = new DataSet(labels.length);
		
		for(int i = 0; i < labels.length; i++) {
			data.addEntry(Matrix.fromArray(toInput(images.get(i))), Matrix.fromArray(toAnswer(labels[i])));
		}
		
		testData = new DataSet(labels2.length);
		
		for(int i = 0; i < labels2.length; i++) {
			testData.addEntry(Matrix.fromArray(toInput(images2.get(i))), Matrix.fromArray(toAnswer(labels2[i])));
		}
		System.out.println("Loaded data!");
	}
	
	public static DataSet getData() {
		return data;
	}
	
	public static DataSet getTestData() {
		return testData;
	}
	
	//Matrix-Objekt fetch
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
 	
	//One-Hot-Encoded-Vektor
	//Am Index, der der Ziffer entspricht wird die Wahrscheinlichkeit auf 1 gesetzt
 	private static double[] toAnswer(int label) {
 		double[] array = new double[] {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
 		array[label] = 1;
 		
 		return array;
 		
 	}
	
}
