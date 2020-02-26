package me.micha.machinelearning.lib.trainingdata;

import me.micha.machinelearning.lib.Global;
import me.micha.machinelearning.lib.Matrix;

public class DataSet {

	private Entry[] entries;
	private int c = 0;
	
	//Datensatz mit fester Gr��e
	public DataSet(int set_size) {
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
		try {
			return entries[index];
		} catch(Exception ex) {
			return null;
		}
	}
	
	public int length() {
		return entries.length;
	}
	
	//Zuf�lliger Eintrag im Datensatz
	public Entry randomData() {
		return getEntry(Global.RAND.nextInt(entries.length));
	}
	
	//Input-Gr��e. Bei Mnist = 784
	public int getInputSize() {
		return entries[0].getInput().rows;
	}
	
	//Output-Gr��e. Bei Mnist = 10 (f�r jede Ziffer eine Node)
	public int getAnswerSize() {
		return entries[0].getAnswer().rows;
	}
	
}
