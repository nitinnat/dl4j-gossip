package peersim.gossip;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.List;
import java.util.Random;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ch.qos.logback.core.net.SyslogOutputStream;
import peersim.config.*;
import peersim.core.CommonState;
import peersim.core.Fallible;
import peersim.core.Network;
import peersim.core.Protocol;

public class CentralizedVPNN {

 	//learning parameters of the Neural Network 
    private static int train_length, test_length;
	public static double learning_rate;
	public static int epochs;
	public static int batch_size;
	public static int initmethod; // Method to initialize the weights of the NN. 0 - Random, 1 - Xavier
	public static int activationmethod; // Activation function of layers of NN. 0 - Identity, 1 - Sigmoid, 2 - RELU, 3 - Tanh 
	public static String resourcepath;
	public static int num_classes;
	public static int num_run;

	
	
	
    private static Logger log = LoggerFactory.getLogger(CentralizedVPNN.class);

    
    private static INDArray SquaredLoss(INDArray predictions, INDArray labels) {
		/*
		 Computes the cross-entropy loss between predictions and labels array.
		 */
		int numRows = predictions.rows();
		double loss = 0.0;
		INDArray batchLossVector = Nd4j.zeros(numRows, 1);
		for(int i=0;i<numRows;i++) {
			
			loss = Math.pow(labels.getDouble(i) - predictions.getDouble(i), 2);
			batchLossVector.putScalar(i, 0, loss);
		}
		return batchLossVector;
	}
    
    private static double computeLoss(DataSet dataset, MultiLayerNetwork model, int start_index, int end_index) {
		/**
		 * Computes the loss on a dataset.
		 */
    	
		start_index = 0;
		end_index = dataset.numExamples();
		
		INDArray features = dataset.getFeatures().get(NDArrayIndex.interval(start_index*batch_size,end_index*batch_size), NDArrayIndex.all());
		INDArray labels = dataset.getLabels().get(NDArrayIndex.interval(start_index*batch_size,end_index*batch_size), NDArrayIndex.all());
		//features = features.div(255);
		model.setInput(features);
		List<INDArray> activations = model.feedForward(true, false);
		INDArray predictions = activations.get(activations.size() - 1);
		INDArray loss_vector = SquaredLoss(predictions, labels);
		double score = loss_vector.sumNumber().doubleValue();
		score /= loss_vector.size(0);
        return score;
		
	}
    
    private static double computeAccuracy(DataSet dataset, MultiLayerNetwork model) {
		/**
		 * Computes the loss on a dataset.
		 */
		INDArray features = dataset.getFeatures();
		INDArray labels = dataset.getLabels();
		List<INDArray> activations = model.feedForward(features);
		INDArray predictions = activations.get(activations.size() - 1);
		
		//TODO: Write a clause for argmax for multiclass
		int correctCount = 0;
		for(int i=0;i<labels.length(); i++) {
			int label = labels.getInt(i);
			int pred = 0;
			if (predictions.getInt(i) >= 0.5){
				pred = 1;
			}
			 if (label == pred) {
				 correctCount += 1;		 
			 }
			 
		}
		double accuracy = correctCount*100/labels.length();
		return accuracy;
		
	}
    
    public static void main(String[] args) throws Exception {
    	
    	String prefix = args[0];
    	System.out.println(prefix);
    	learning_rate = 0.1;
    	batch_size = 32;
    	epochs = 10000;
    	
    	initmethod = 0;
    	activationmethod = 0;
    	train_length = 60000;
    	test_length = 10000;
    	num_classes = 1;
    	num_run = 0;
    	
    	
    	long startTime = System.nanoTime();
    	// Determine base dataset name
    	String resourcepath = "data/synthetic";
		String[] temp_data = resourcepath.split("/");
		String base_dataset_name = temp_data[temp_data.length - 1];
		
    	// Get train file and test file paths
        String localTrainFilepath = resourcepath + "/" + base_dataset_name + "_train_binary.csv";
        String localTestFilepath = resourcepath + "/" + base_dataset_name + "_test_binary.csv";
        System.out.println(localTrainFilepath);
    	// Load the entire train set to compute train loss
        DataSetIterator trainIter = null;
    	DataSetIterator testIter = null;
    	DataSet trainSet;
    	DataSet testSet;
    	
        System.out.println("Reading training file "+ localTrainFilepath);
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(localTrainFilepath)));
        trainIter = new RecordReaderDataSetIterator(rr,train_length,0,num_classes);
        trainSet = trainIter.next();
        System.out.println(trainSet.get(0));
        
        //Load the entire test/evaluation set to compute test loss
        System.out.println("Reading file "+ localTestFilepath);
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File(localTestFilepath)));
        testIter = new RecordReaderDataSetIterator(rrTest,test_length,0,num_classes);
        testSet = testIter.next();
        double readingTime = (double) ((System.nanoTime() - startTime)/1e-9);
     // Create Neural Network
        int numInputs = (int)trainSet.getFeatures().size(1);
        int numOutputs = num_classes;
        int numHiddenNodes = 20;
        int nIn = numInputs;
        int nOut = numOutputs;
        WeightInit initMethod = WeightInit.XAVIER;
        if (initmethod == 1) {
        	initMethod = WeightInit.XAVIER;
        }
        Activation activationMethod = null;
        switch (activationmethod) {
        case 0: activationMethod = Activation.IDENTITY;
        case 1: activationMethod = Activation.SIGMOID;
        case 2: activationMethod = Activation.RELU;
        case 3: activationMethod = Activation.TANH;
        default: activationMethod = Activation.IDENTITY;
        };
        //TODO: Write a similar switch-case for Updater.
        System.out.println("Creating a neural network of " + nIn + 
        		" inputs, " + "1 hidden layer of " + numHiddenNodes + " and " + nOut + " output nodes.");
        
        Random rand = new Random();

	     // Obtain a number between [0 - 49].
	     int seed = rand.nextInt(5000);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(1234)
            .weightInit(initMethod)
            .updater(new Sgd())
            .list()
            .layer(0, new DenseLayer.Builder().nIn(nIn).nOut(numHiddenNodes)
            		
            		.activation(Activation.RELU)
            		.build())
            .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(nOut)
            		.activation(Activation.SIGMOID)
            		.build())
            .backprop(true).pretrain(false)
            .build();


	        MultiLayerNetwork model = new MultiLayerNetwork(conf);
	        model.init();
        
	     // Create headers to store the results
		String csv_filename = resourcepath + "/run" + num_run + "/centralized_nn_results.csv";
		System.out.println("Storing in " + csv_filename);
		String opString = "Iter,TrainLoss,TrainAccuracy,TestLoss,TestAccuracy,ReadingTime,TrainTime";
		
		// Write to file
		try {
		BufferedWriter bw = new BufferedWriter(new FileWriter(csv_filename));
		
		bw.write(opString);
		bw.write("\n");
		bw.close();
		}
		catch(Exception e) {
			
		}
        log.info("Train model....");
        
        
        
        double trainTime = 0.0;
        int start_index;
        int end_index;
        INDArray cur_features = trainSet.getFeatures();
		int num_batches = (int)cur_features.size(0)/batch_size;
		
	
        
        for( int iter=0; iter<epochs; iter++ ){
        	startTime = System.nanoTime();
        	System.out.println("Epoch " + iter);
        	start_index = iter%num_batches;
    		end_index = start_index+1;
            //INDArray features = trainIter.next().getFeatures();
        	INDArray features = trainSet.getFeatures().get(NDArrayIndex.interval(start_index*batch_size,end_index*batch_size), NDArrayIndex.all());
    		//features = features.div(255);
        	INDArray labels = trainSet.getLabels().get(NDArrayIndex.interval(start_index*batch_size,end_index*batch_size), NDArrayIndex.all());
			
    		model.setInput(features);
			List<INDArray> activations = model.feedForward(true, false);
			INDArray predictions = activations.get(activations.size() - 1);
			INDArray squared_loss = SquaredLoss(predictions, labels);
			double score = squared_loss.sumNumber().doubleValue();
			score /= squared_loss.size(0);
			System.out.println("LOSS: "+score);
			System.out.println("Loss: " + model.output(testSet.getFeatures()));
			int iteration = 0;
	        int epoch = 0;
	        model.setInput(features);
			Pair<Gradient, INDArray> p = model.backpropGradient(squared_loss, null);  //Calculate backprop gradient based on error array
			Gradient gradient_pn = p.getFirst();
			//Update the gradient: apply learning rate, momentum, etc
	        //This modifies the Gradient object in-place
	        model.getUpdater().update(model, gradient_pn, iteration, epoch, batch_size, LayerWorkspaceMgr.noWorkspaces());
	        //Get a row vector gradient array, and apply it to the parameters to update the model
	        INDArray update_vector_pn = gradient_pn.gradient();
	        model.params().subi(update_vector_pn);
	        
	        trainTime += (double)((System.nanoTime() - startTime)/1e-9);
	        
	     // Compute loss on the entire test set for all nodes
	        double test_loss = 0;
	        double train_loss = 0;
	        double train_accuracy = 0;
	        double test_accuracy = 0;
	        /*
	        if ((iter%10) == 0) {
	        	
	        	// Compute loss on train batch
	        	System.out.println(trainSet.getFeatures().shapeInfoToString());
	        	System.out.println(testSet.getFeatures().shapeInfoToString());
	        	train_loss = computeLoss(trainSet, model, start_index, end_index);
		        System.out.println("Train loss after iteration " + iter + " is:" + train_loss);
				train_accuracy = computeAccuracy(trainSet, model);
				System.out.println("Train accuracy after iteration " + iter + " is:" + train_accuracy);
		        test_loss = computeLoss(testSet, model, start_index, end_index);
				System.out.println("Mean test loss after iteration " + iter + " is:" + test_loss);
				test_accuracy = computeAccuracy(testSet, model);
				System.out.println("Test accuracy after iteration " + iter + " is:" + test_accuracy);
				
				
	        }
	        
	        */
			
			opString = iter + "," + train_loss +"," + train_accuracy + ","+ test_loss +","+ test_accuracy +","+readingTime + "," + trainTime;
			
			// Write to file
			try {
			BufferedWriter bw = new BufferedWriter(new FileWriter(csv_filename, true));
			
			bw.write(opString);
			bw.write("\n");
			bw.close();
			}
			catch(Exception e) {
				
			}
        }


       

    }
    

}