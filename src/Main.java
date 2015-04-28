
public class Main {
	
	public static void main(String[] args) {
		ClassifyThis bc = new ClassifyThis("census.names");
		bc.train("census.train");
		bc.makePredictions("census.test");
	}
}
