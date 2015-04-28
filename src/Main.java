
public class Main {
	
	public static void main(String[] args) {
		BAClassifier bc = new BAClassifier("String to file");
		bc.train("Training File");
		bc.makePredictions("Prediction File");
	}

}
