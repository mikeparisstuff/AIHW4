
public class Main {
	
	public static void main(String[] args) {
		BAClassifier bc = new BAClassifier("/Users/MichaelParis/uva/Spring 2015/AI/HW4/trainingData/census.names");
		bc.train("/Users/MichaelParis/uva/Spring 2015/AI/HW4/trainingData/census.train");
		bc.makePredictions("/Users/MichaelParis/uva/Spring 2015/AI/HW4/trainingData/census.test");
	}

}
