/*
 * Peersim-Gadget : A Gadget protocol implementation in peersim based on the paper
 * Chase Henzel, Haimonti Dutta
 * GADGET SVM: A Gossip-bAseD sub-GradiEnT SVM Solver   
 * 
 * Copyright (C) 2012
 * Deepak Nayak 
 * Columbia University, Computer Science MS'13
 * 
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 * 
 * You should have received a copy of the GNU General Public License along with
 * this program; if not, write to the Free Software Foundation, Inc., 51
 * Franklin St, Fifth Floor, Boston, MA 02110-1301 USA
 */
package peersim.gossip;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.File;
import java.io.IOException;
import java.io.LineNumberReader;
import java.net.MalformedURLException;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import peersim.config.*;
import peersim.core.*;

/**
 * Class PegasosNode
 * An implementation of {@link Node} which can handle external resources. 
 * It is based on {@link GeneralNode} and extended it to be used with pegasos solver.
 * At the start of node, each node has some associated data file, on which it calls
 * training function through a jni call. 
 * It will take the resourcepath from config file.
 * This is another implementation of {@link Node} class that is used to compose the
 * p2p {@link Network}, where each node can handle an external resource.
 * @author Nitin Nataraj
 */
public class PegasosNode implements Node {

	// ================= fields ========================================
	// =================================================================

	/**
	 * New config option added to get the resourcepath where the resource file
	 * should be generated. Resource files are named <ID> in resourcepath.
	 * @config
	 */

	/**
	 * New config options added to set the learning parameters of pegasos
	 * PAR_LAMBDA 	: lambda parameter, default value 0.001 
	 * PAR_MAXITER 	: maximum number of iteration, defaults to 100000
	 * PAR_EXAM_PER_ITER 	: number of examples to consider for stochastic
	 * gradient computation, defaults to 1
	 * @config
	 */
	
	// Neural net params in the config file
	private static final String PAR_LEARNING_RATE = "learningrate";
	private static final String PAR_BATCH_SIZE = "batchsize";
	private static final String PAR_INITIALIZATION = "initmethod";
	private static final String PAR_ACTIVATION_METHOD = "activation";
	private static final String PAR_EPOCHS = "epochs";
	private static final String PAR_TRAINLEN = "trainlen";
	private static final String PAR_TESTLEN = "testlen";
	private static final String PAR_NUMCLASSES = "numclasses";
	private static final String PAR_PATH = "resourcepath";	
	private static final String PAR_SIZE = "size";
	private static long counterID = -1; // used to generate unique IDs 
	protected Protocol[] protocol = null; //The protocols on this node.
	
 	//learning parameters of the Neural Network 
    private int train_length, test_length;
	public double learning_rate;
	public int epochs;
	public int batch_size;
	public int initmethod; // Method to initialize the weights of the NN. 0 - Random, 1 - Xavier
	public int activationmethod; // Activation function of layers of NN. 0 - Identity, 1 - Sigmoid, 2 - RELU, 3 - Tanh 
	/**
	 * The current index of this node in the node
	 * list of the {@link Network}. It can change any time.
	 * This is necessary to allow
	 * the implementation of efficient graph algorithms.
	 */
	private int index;

	/**
	 * The fail state of the node.
	 */
	protected int failstate = Fallible.OK;

	/**
	 * The ID of the node. It should be final, however it can't be final because
	 * clone must be able to set it.
	 */
	private long ID;

	/**
	 * The prefix for the resources file. All the resources file will be in prefix 
	 * directory. later it should be taken from configuration file.
	 */
	public String resourcepath;
	private int numNodes;
	public int numRun;
	public boolean converged = false;
	public long num_features;
	MultiLayerNetwork model = null;
	DataSetIterator trainIter = null;
	DataSetIterator testIter = null;
	public DataSet trainSet;
	public DataSet testSet;
	public int convergedCount = 0;
	public double epsilon = 0.001;
	public double prevLoss = 0;
	public int num_classes;
	public double readTime = 0.0;
	public double trainTime = 0.0;
	public double gossipTime = 0.0;
	
	
	// ================ constructor and initialization =================
	// =================================================================
	/** Used to construct the prototype node. This class currently does not
	 * have specific configuration parameters and so the parameter
	 * <code>prefix</code> is not used. It reads the protocol components
	 * (components that have type {@value peersim.core.Node#PAR_PROT}) from
	 * the configuration.
	 */

	
	public PegasosNode(String prefix) {
		System.out.println(prefix);
		String[] names = Configuration.getNames(PAR_PROT);
		resourcepath = (String)Configuration.getString(prefix + "." + PAR_PATH);
		learning_rate = Configuration.getDouble(prefix + "." + PAR_LEARNING_RATE, 1e-6);
		batch_size = Configuration.getInt(prefix + "." + PAR_BATCH_SIZE, 4);
		epochs = Configuration.getInt(prefix + "." + PAR_EPOCHS, 1);
		initmethod = Configuration.getInt(prefix + "." + PAR_INITIALIZATION, 1);
		activationmethod = Configuration.getInt(prefix + "." + PAR_ACTIVATION_METHOD, 0);
		train_length = Configuration.getInt(prefix + "." + PAR_TRAINLEN, 0);
		test_length = Configuration.getInt(prefix + "." + PAR_TESTLEN, 0);
		num_classes = Configuration.getInt(prefix + "." + PAR_NUMCLASSES, 1);
		
		
		System.out.println("model file and train file are saved in: " + resourcepath);
		CommonState.setNode(this);
		ID = nextID();
		protocol = new Protocol[names.length];
		for (int i=0; i < names.length; i++) {
			CommonState.setPid(i);
			Protocol p = (Protocol) 
					Configuration.getInstance(names[i]);
			protocol[i] = p; 
		}
		numNodes = Configuration.getInt(prefix + "." + PAR_SIZE, 20);
		numRun = Configuration.getInt(prefix + "." + "run", 0);
		System.out.println("Number of nodes is ####### "+numNodes);
	}
	
	/**
	 * Used to create actual Node by calling clone() on a prototype node. So, actually 
	 * a Node constructor is only called once to create a prototype node and after that
	 * all nodes are created by cloning it.
	 
	 */
	
	public Object clone() {
		
		
		PegasosNode result = null;
		
		try { 
			result=(PegasosNode)super.clone(); 
			}
		catch( CloneNotSupportedException e ) {} // never happens
		
		result.protocol = new Protocol[protocol.length];
		CommonState.setNode(result);
		result.ID = nextID();
		for(int i=0; i<protocol.length; ++i) {
			CommonState.setPid(i);
			result.protocol[i] = (Protocol)protocol[i].clone();
		}
		System.out.println("creating node with ID: " + result.getID());
		
		// Determine base dataset name
		String[] temp_data = resourcepath.split("/");
		String base_dataset_name = temp_data[temp_data.length - 1];
        
		// Get train file and test file paths
        String localTrainFilepath = resourcepath + "/" + base_dataset_name + "_train_" + result.getID() + ".csv";
        String localTestFilepath = resourcepath + "/" + base_dataset_name + "_test_" + result.getID() + ".csv";
        
      
		// Create a folder for this run if it does not exist
		File directory = new File(resourcepath + "/run" + result.numRun);
	    if (! directory.exists()){
	        directory.mkdir();
	    
	    }
	    
	    
	    if (result.getID() == 0) {
	    	// Create headers to store the results
	    	System.out.println("Creating csv file to store the results.");
	    	String csv_filename = resourcepath + "/run" + result.numRun + "/vpnn_results.csv";
			System.out.println("Storing in " + csv_filename);
			String opString = "Node,Iter,TrainLoss,TestLoss,TrainTime,ReadTime";
			
			// Write to file
			try {
			BufferedWriter bw = new BufferedWriter(new FileWriter(csv_filename));
			
			bw.write(opString);
			bw.write("\n");
			bw.close();
			}
			catch(Exception e) {
				
			}
	    }
		
		
		long startTime, endTime;
		
		try {
			    startTime = System.nanoTime();    
		        
			    // Load the entire train set
		        System.out.println("Reading training file "+ localTrainFilepath);
		        RecordReader rr = new CSVRecordReader();
		        rr.initialize(new FileSplit(new File(localTrainFilepath)));
		        result.trainIter = new RecordReaderDataSetIterator(rr,train_length,0,num_classes);
		        result.trainSet = result.trainIter.next();
		        result.trainSet.shuffle(12345);
		        
		        //Load the entire test/evaluation set
		        System.out.println("Reading test file "+ localTestFilepath);
		        RecordReader rrTest = new CSVRecordReader();
		        rrTest.initialize(new FileSplit(new File(localTestFilepath)));
		        result.testIter = new RecordReaderDataSetIterator(rrTest,test_length,0,num_classes);
		        result.testSet = result.testIter.next();
		        result.testSet.shuffle(12345);
		        
		        endTime = System.nanoTime();
		        result.readTime = (double)(endTime - startTime)/1e-9;
		        // Create Neural Network
		        int numInputs = (int)result.trainSet.getFeatures().size(1);
		        int numOutputs = num_classes;
		        int numHiddenNodes = 40;
		        int nIn = numInputs;
		        int nOut = numOutputs;
		        
		        WeightInit initMethod = WeightInit.XAVIER;
		        if (initmethod == 1) {
		        	initMethod = WeightInit.XAVIER;
		        }
		        if (initmethod == 2) {
		        	initMethod = WeightInit.NORMAL;
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

		        
		        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
		            .seed(123*result.getID()) // Setting the same initialization seed for all networks.
		            .weightInit(WeightInit.XAVIER)
		            .updater(new Sgd(0.01))
		            .list()
		            .layer(0, new DenseLayer.Builder().nIn(nIn).nOut(numHiddenNodes)
		            		
		            		.activation(Activation.RELU)
		            		.build())
		            .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(nOut)
		            		.activation(Activation.SIGMOID)
		            		.build())
		            .backprop(true).pretrain(false)
		            .build();

		        result.model = new MultiLayerNetwork(conf);
		        result.model.init();
		        result.num_features = numInputs;
		        
		        
		        //writeParamsToFile(result.model.params());
		        /*
		        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
		                .seed(seed)
		                .updater(new Nesterovs(learningRate, 0.9))
		                .list()
		                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
		                        .weightInit(WeightInit.XAVIER)
		                        .activation(Activation.RELU)
		                        .build())
		                .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
		                        .weightInit(WeightInit.XAVIER)
		                        .activation(Activation.SOFTMAX)
		                        .nIn(numHiddenNodes).nOut(numOutputs).build())
		                .backprop(true).pretrain(false)
		                .build();


		        result.model = new MultiLayerNetwork(conf);
		        result.model.init();
		        */
		        result.model.setListeners(new ScoreIterationListener(10));  //Print score every 10 parameter updates
		        System.out.println("Parameters of the model");
		        System.out.println(result.model.params().toString());
			//readInitTime = System.nanoTime() - startTime;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
				return result;
		

	}


	private void writeParamsToFile(String filename, INDArray params, int cycle) {
		// TODO Auto-generated method stub
		try {
			BufferedWriter bw = new BufferedWriter(new FileWriter(filename));
			bw.write("Cycle " + cycle);
			bw.write("----------------------------------------------------");
			
			bw.close();
			}
			catch(Exception e) {
				
			}
	}

 
	/** returns the next unique ID */
	private long nextID() {

		return counterID++;
	}

	// =============== public methods ==================================
	// =================================================================


	public void setFailState(int failState) {

		// after a node is dead, all operations on it are errors by definition
		if(failstate==DEAD && failState!=DEAD) throw new IllegalStateException(
				"Cannot change fail state: node is already DEAD");
		switch(failState)
		{
		case OK:
			failstate=OK;
			break;
		case DEAD:
			//protocol = null;
			index = -1;
			failstate = DEAD;
			for(int i=0;i<protocol.length;++i)
				if(protocol[i] instanceof Cleanable)
					((Cleanable)protocol[i]).onKill();
			break;
		case DOWN:
			failstate = DOWN;
			break;
		default:
			throw new IllegalArgumentException(
					"failState="+failState);
		}
	}

	public int getFailState() { return failstate; }

	public boolean isUp() { return failstate==OK; }

	public Protocol getProtocol(int i) { return protocol[i]; }

	public int protocolSize() { return protocol.length; }

	public int getIndex() { return index; }

	public void setIndex(int index) { this.index = index; }
        
        
	/**
	 * Returns the ID of this node. The IDs are generated using a counter
	 * (i.e. they are not random).
	 */
	public long getID() { return ID; }

	public String toString() 
	{
		StringBuffer buffer = new StringBuffer();
		buffer.append("ID: "+ID+" index: "+index+"\n");
		for(int i=0; i<protocol.length; ++i)
		{
			buffer.append("protocol[" + i +"]=" + protocol[i] + "\n");
		}
		return buffer.toString();
	}

	/** Implemented as <code>(int)getID()</code>. */
	public int hashCode() { return (int)getID(); }

	public static void main(String[] args) {
		PegasosNode result = new PegasosNode("config/config-pegasosReuters.cfg");
		
	}

}
