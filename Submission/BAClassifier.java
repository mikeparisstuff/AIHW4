import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import Jama.Matrix;


public class BAClassifier extends Classifier{
	
	// List contains names of all attributes in order.
	List<String> valNames = new ArrayList<String>(0);
	// Hashmap contains mapping name -> value options
	// List is empty if the value is numeric, else the list of options
	HashMap<String, List<String>> valTypes = new HashMap<String, List<String>>();
	
	List<String> trainingData = new ArrayList<String>();
	
	Matrix attrMat;
	Matrix outputMat;
	Matrix thetas;
	
	List<String> outputs;
	List<String> trainingOutputs = new ArrayList<String>();
	List<String> predictions = new ArrayList<String>();

	public BAClassifier(String namesFilepath) {
		super(namesFilepath);
		try {
			BufferedReader reader = new BufferedReader(new FileReader(namesFilepath));
			String line;
			int lineNum = 0;
			while ((line = reader.readLine()) != null) {
				// line holds entry.
				if (lineNum == 0) {
					// These are potential outputs.
					outputs = Arrays.asList(line.split("\\s+"));
				} else {
					// If line is not empty
					if (line.length() != 0) {
						// If line is of length 2 we have a numeric
						String[] vals = line.split("\\s+");
						// Add the val name to the valNames array
						valNames.add(vals[0]);
						if (vals.length == 2) {
							// Just make sure its still numeric
							if (vals[1].equals("numeric")) {
								valTypes.put(vals[0], new ArrayList<String>());
							}
						} else {
							// This is a discrete valued attribute
							String[] attrs = Arrays.copyOfRange(vals, 1, vals.length);
							valTypes.put(vals[0], new ArrayList<String>(Arrays.asList(attrs)));
						}
					}
				}
				lineNum++;
			}
		}
		catch (Exception e) {
			System.err.format("Exception occured trying to read '%s'.", namesFilepath);
		}
//		System.out.println("Finished Reading File");
		for (int i = 0; i < valNames.size(); i++) {
			List<String> vals = valTypes.get(valNames.get(i));
//			System.out.format("Vals for '%s': %s\n", valNames.get(i), vals.toString());
		}
		
	}

	public void train(String trainingDataFilpath) {
//		System.out.println("Training");
		
		try {
			BufferedReader reader = new BufferedReader(new FileReader(trainingDataFilpath));
			String line;
			while ((line = reader.readLine()) != null) {
				// Start building the matrix
//					double[] row = new double[valNames.size()];
				trainingData.add("1 " + line);
			}
		}
		catch (Exception e) {
			System.err.format("Exception occured trying to read '%s'.", trainingDataFilpath);
		}
		int lineNum = trainingData.size();
		
		double[][] allData = new double[lineNum][valNames.size()];
		double[][] outputData = new double[lineNum][1];
		
//		List<Double> allData = new ArrayList<Double>();
//		List<Double> outputData = new ArrayList<Double>();
//		while ((line = reader.readLine()) != null) {
		for (int l = 0; l < lineNum; l++) {
			
			// Start building the matrix

			String[] vals = trainingData.get(l).split("\\s+");
			double[] row = new double[vals.length-1];
			int i = 0;
			for (; i < vals.length-1; i++) {
				// we are looking at this attr
				if ( i == 0) {
					row[i] = Double.parseDouble(vals[i]);
					continue;
				}
				String attr = valNames.get(i-1);
				List<String> discVals = valTypes.get(attr);
				if (discVals.size() == 0) {
					// We are looking at a numeric
					row[i] = Double.parseDouble(vals[i]);
				} else {
					// Find the index of vals[i] in discVals and assign that integer value
					row[i] = (double)discVals.indexOf(vals[i]);
				}
			}
			allData[l] = row;
			// add outputs value
			double[] v = new double[1];
			trainingOutputs.add(vals[i]);
			v[0] = (double)outputs.indexOf(vals[i]);
			outputData[l] =  v ;
//			outputData.add((double)outputs.indexOf(vals[i]));
			
		}

//		System.out.println("Done");
		Matrix m = new Matrix(allData);
		Matrix outputM = new Matrix(outputData);
		attrMat = m;
		outputMat = outputM;
		thetas = calculateTheta();
		
		
	}
	//  𝜃 = (𝑋𝑇𝑋)−1 ∗ 𝑋𝑇Y
	private Matrix calculateTheta() {
		Matrix tran1 = attrMat.transpose();
		Matrix left = tran1.times(attrMat).inverse();
		Matrix right = tran1.times(outputMat);
		return left.times(right);
	}

	public void makePredictions(String testDataFilepath) {
//		System.out.println("Making Predictions");
		try {
			BufferedReader reader = new BufferedReader(new FileReader(testDataFilepath));
			String line;
			int lineNum = 0;
			while ((line = reader.readLine()) != null) {
				// Start building the matrix
//				double[] row = new double[valNames.size()];
				String[] vals = line.split("\\s+");
				double sum = thetas.get(0,0);
				for (int i = 0; i < vals.length; i++) {
					String attr = valNames.get(i);
					List<String> discVals = valTypes.get(attr);
					if (discVals.size() == 0) {
						// We are looking at a numeric
//						System.out.println("Adding value to sum: " + thetas.get(i+1, 0));
						sum += thetas.get(i+1, 0) * Double.parseDouble(vals[i]);
					} else {
						// Find the index of vals[i] in discVals and assign that integer value
//						System.out.println("Adding value to sum: " + thetas.get(i+1, 0));
						sum += thetas.get(i+1, 0) * ((Number)discVals.indexOf(vals[i])).doubleValue(); 
//						row.add(((Number)discVals.indexOf(vals[i])).doubleValue());
					}
				}
				int ind = (sum >= .5) ? 1 : 0;
//				System.out.format("Prediction: '%s'\n", outputs.get(ind));
				System.out.println(outputs.get(ind));
				predictions.add(outputs.get(ind)+";"+sum);
				lineNum++;
			}
		}
		catch (Exception e) {
			System.err.format("Exception occured trying to read '%s'.", testDataFilepath);
		}
//		checkPredictions();
//		System.out.println(predictions.size());
	}
	
	private void checkPredictions() {
		int numCorrect = 0;
		int numWrong = 0;
		for (int i = 0; i < predictions.size(); i++) {
			if(predictions.get(i).split(";")[0].equals(trainingOutputs.get(i))) {
				numCorrect++;
			} else {
				numWrong++;
			}
		}
		System.out.format("Num Correct: %d\nNum Wrong: %d", numCorrect, numWrong);
//		String csv = toCSV();
//		System.out.print(csv);
	}
	
	private String toCSV() {
		StringBuilder b = new StringBuilder();
		for(int i = 0; i < predictions.size(); i++) {
			b.append(predictions.get(i).split(";")[0] + ";" + predictions.get(i).split(";")[1] + ";\n" + trainingOutputs.get(i));
			String line;
			if (predictions.get(i).split(";")[0].equals(trainingOutputs.get(i))) {
				line = predictions.get(i).split(";")[0] + ";" + predictions.get(i).split(";")[1] + ";" + trainingOutputs.get(i) + ";" + (double)outputs.indexOf(trainingOutputs.get(i)) + ";correct;1";
			} else {
				line = predictions.get(i).split(";")[0] + ";" + predictions.get(i).split(";")[1] + ";" + trainingOutputs.get(i) + ";" + (double)outputs.indexOf(trainingOutputs.get(i)) + ";incorrect;0";
			}
			System.out.println(line);
		}
		return b.toString();
//		System.out.println("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
	}
	
}
