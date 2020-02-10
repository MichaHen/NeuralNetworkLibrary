package me.micha.machinelearning;

import java.util.Random;

public class TrainingData {

	private Entry[] entries;
	private int c = 0;
	private Random rand;
	
	public TrainingData(int set_size) {
		entries = new Entry[set_size];
	}
	
	public void addEntry(Matrix input, Matrix answer) {
		if(c < entries.length) {
			entries[c] = new Entry(input, answer);
			c++;
		}
	}
	
	public Entry[] getEntries() {
		return entries;
	}
	
	public Entry getEntry(int index) {
		return entries[index];
	}
	
	public int length() {
		return entries.length;
	}
	
	public Entry randomData() {
		if(rand == null) rand = new Random();
		
		return getEntry(rand.nextInt(entries.length));
	}
	
	public int getInputSize() {
		return entries[0].getInput().rows;
	}
	
	public int getAnswerSize() {
		return entries[0].getAnswer().rows;
	}
	
}
