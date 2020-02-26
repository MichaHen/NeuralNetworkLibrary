package me.micha.machinelearning.lib.calc;

import me.micha.machinelearning.lib.NN;
import me.micha.machinelearning.lib.mnist.MnistConverter;

//Thread der ein Netzwerk für gewuenschte Parameter trainiert => Zur Berechnung der Durchschnittswerte von gleichen NN mit untersch. initialisierten Gewichten
public class NNExecutor extends Thread {

	NN dnn;
	int EPOCHS, batchSize, threads;
	double[] rates;
	
	public NNExecutor(NN dnn, int EPOCHS, int batchSize, int threads) {
		this.dnn = dnn;
		this.EPOCHS = EPOCHS;
		this.rates = new double[EPOCHS+1];
		this.threads = threads;
	}
	
	@Override
	public void run() {
		for(int i = 0; i <= EPOCHS; i++) {
			rates[i] = dnn.test(MnistConverter.getTestData(), threads);
			System.out.println("EPOCH " + i);
			System.out.println(rates[i]);
			
			long start = System.currentTimeMillis();
			if(i < EPOCHS) {
				dnn.trainThreadPooled(MnistConverter.getData(), batchSize, threads);
				System.out.println("T: " + ((System.currentTimeMillis() - start)/1000) + "s");
			}
		}
	}
	
	public double[] getRates() {
		return rates;
	}
	
	
}
