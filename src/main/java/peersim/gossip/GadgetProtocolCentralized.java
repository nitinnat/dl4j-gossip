/*
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

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.primitives.Pair;

import java.util.Arrays;
import java.lang.Integer;

import java.io.FileReader;

import java.io.LineNumberReader;
import peersim.gossip.PegasosNode;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import peersim.config.Configuration;
import peersim.config.FastConfig;
import peersim.core.*;
import peersim.cdsim.*;


import java.net.MalformedURLException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.GeneralSecurityException;
import java.text.ParseException;
import java.io.BufferedReader;

/**
 *  @author Nitin Nataraj
 */


public class GadgetProtocolCentralized implements CDProtocol {
	private static final String PAR_LAMBDA = "lambda";
	private static final String PAR_ITERATION = "iter";
	public static boolean flag = false;
	public static int t = 0;
	public static boolean optimizationDone = false;	
	public double EPSILON_VAL = 0.01;
	protected int lid;
	protected double lambda;
	protected int T;
	public static double[][] optimalB;
	public static int end = 0;
	public static boolean pushsumobserverflag = false;
	public static final int CONVERGENCE_COUNT = 10;
	private String protocol;
	private String resourcepath;


	/**
	 * Default constructor for configurable objects.
	 */
	public GadgetProtocolCentralized(String prefix) {
		lid = FastConfig.getLinkable(CommonState.getPid());
		
		protocol = Configuration.getString(prefix + "." + "prot", "pushsum1");
		
	}

	/**
	 * Returns true if it possible to deliver a response to the specified node,
	 * false otherwise.
	 * Currently only checking failstate, but later may be we need to check
	 * the non-zero transition probability also
	 */
	protected boolean canDeliverRequest(Node node) {
		if (node.getFailState() == Fallible.DEAD)
			return false;
		return true;
	}
	/**
	 * Returns true if it possible to deliver a response to the specified node,
	 * false otherwise.
	 * Currently only checking failstate, but later may be we need to check
	 * the non-zero transition probability also
	 */
	protected boolean canDeliverResponse(Node node) {
		if (node.getFailState() == Fallible.DEAD)
			return false;
		return true;
	}

	/**
	 * Clone an existing instance. The clone is considered 
	 * new, so it cannot participate in the aggregation protocol.
	 */
	public Object clone() {
		GadgetProtocolCentralized gp = null;
		try { gp = (GadgetProtocolCentralized)super.clone(); }
		catch( CloneNotSupportedException e ) {} // never happens
		return gp;
	}
	
	/**
	 * Computes the average loss vector using the networks present on two nodes 
	 * and then backpropagates the same error back in both networks.
	 * 
	 */

	private void pushLossVectorSquaredLoss(PegasosNode pn, PegasosNode peer, int start_index, int end_index) {
		
		
		// Compute the features and labels for the nodes corresponding to the current batch
		INDArray features_pn = pn.trainSet.getFeatures().get(NDArrayIndex.interval(start_index*pn.batch_size,end_index*pn.batch_size), NDArrayIndex.all());
		INDArray features_peer = peer.trainSet.getFeatures().get(NDArrayIndex.interval(start_index*pn.batch_size,end_index*pn.batch_size), NDArrayIndex.all());
		INDArray labels_pn = pn.trainSet.getLabels().get(NDArrayIndex.interval(start_index*pn.batch_size,end_index*pn.batch_size), NDArrayIndex.all());
		INDArray labels_peer = peer.trainSet.getLabels().get(NDArrayIndex.interval(start_index*pn.batch_size,end_index*pn.batch_size), NDArrayIndex.all());;
		assert labels_pn.equals(labels_peer) : "There's a data mismatch, check data.";
		
		// Set input on both models
		pn.model.setInput(features_pn);
		peer.model.setInput(features_peer);
		
		// Determine activations of the last layer after going through the feedforward phase
		List<INDArray> activations_pn = pn.model.feedForward(true, false);
		List<INDArray> activations_peer = peer.model.feedForward(true, false);
		INDArray predictions_pn = activations_pn.get(activations_pn.size() - 1);
		INDArray predictions_peer = activations_peer.get(activations_peer.size() - 1);
		
		// Compute squared loss on both nodes, and then calculate the average loss vector.
		INDArray squared_loss_pn = crossEntropyLoss(predictions_pn, labels_pn);
		INDArray squared_loss_peer = crossEntropyLoss(predictions_peer, labels_peer);
		//System.out.println("Squared Loss " + pn.getID());
		//System.out.println(squared_loss_pn);
		//System.out.println("Squared Loss " + peer.getID());
		//System.out.println(squared_loss_peer);
		INDArray average_loss = squared_loss_pn.add(squared_loss_peer).div(2);
		//System.out.println("Average Loss");
		//System.out.println(average_loss);
		//System.out.println("Average Loss " + squared_loss_pn);
		// Dummy variables needed for DL4J model updates
		int iteration = 0;
        int epoch = 0;
		
        // Obtain the gradient after backpropagation
		Pair<Gradient, INDArray> p_pn = pn.model.backpropGradient(average_loss, null);  //Calculate backprop gradient based on error array
		Gradient gradient_pn = p_pn.getFirst();
		
		//Update the gradient: apply learning rate, momentum, etc
        //This modifies the Gradient object in-place
        pn.model.getUpdater().update(pn.model, gradient_pn, iteration, epoch, pn.batch_size, LayerWorkspaceMgr.noWorkspaces());
        INDArray update_vector_pn = gradient_pn.gradient();
        pn.model.params().subi(update_vector_pn);
        
        // Update the params of the peer node as well
		Pair<Gradient, INDArray> p_peer = peer.model.backpropGradient(average_loss, null);  //Calculate backprop gradient based on error array
		Gradient gradient_peer = p_peer.getFirst();
        peer.model.getUpdater().update(peer.model, gradient_peer, iteration, epoch, peer.batch_size, LayerWorkspaceMgr.noWorkspaces());
        INDArray update_vector_peer = gradient_peer.gradient();
        peer.model.params().subi(update_vector_peer);
        
        // Debug prints
        //System.out.println(pn.model.params());
        //System.out.println(peer.model.params());
        //System.out.println(SimpleArraySimilarity(pn.model.params(), peer.model.params()));
        	   

	}
	private double SimpleArraySimilarity(INDArray arr1, INDArray arr2) {
		// Computes a simple similarity between two INDArrays by checking if 
		// elements are the same or different
		double similarity = 0.0;
		assert arr1.length() == arr2.length();
		for( int i=0;i<arr1.length();i++) {
			if(arr1.getFloat(i) == arr2.getFloat(i)) {
				similarity += 1;
			}
		}
		
		
		return similarity*100/arr1.length();
		
	}
	private INDArray SquaredLoss(INDArray predictions, INDArray labels) {
		/*
		 Computes the squared loss between predictions and labels array.
		 */
		int numRows = predictions.rows();
		double loss = 0.0;
		INDArray batchLossVector = Nd4j.zeros(numRows, 1);
		for(int i=0;i<numRows;i++) {
			
			loss = Math.pow(labels.getDouble(i) - predictions.getDouble(i), 2)/2;
			batchLossVector.putScalar(i, 0, loss);
		}
		return batchLossVector;
	}

	
	private INDArray crossEntropyLoss(INDArray predictions, INDArray labels) {

		/*
		 Computes the cross-entropy loss between predictions and labels array.
		 */
		int numRows = predictions.rows();
		int numCols = predictions.columns();
		//System.out.println(numRows+ " " + numCols);
		INDArray batchLossVector = Nd4j.zeros(numRows, 1);
		for(int i=0;i<numRows;i++) {
			double loss = 0.0;
			for(int j=0;j<numCols;j++) {
				loss += (labels.getDouble(i,j)) * Math.log(predictions.getDouble(i,j) + 1e-15);
				
			}
			batchLossVector.putScalar(i, 0, -loss);
		}
		return batchLossVector;
	}
	

	
	protected List<Node> getPeers(Node node) {
		Linkable linkable = (Linkable) node.getProtocol(lid);
		if (linkable.degree() > 0) {
			List<Node> l = new ArrayList<Node>(linkable.degree());			
			for(int i=0;i<linkable.degree();i++) {
				l.add(linkable.getNeighbor(i));
			}
			return l;
		}
		else
			return null;						
	}			


	public void nextCycle(Node node, int pid) {
		
		int iter = CDState.getCycle(); // Gets the current cycle of Gadget
		PegasosNode pn = (PegasosNode)node; // Initializes the Pegasos Node
		
		final String resourcepath = pn.resourcepath;
		long startTime = System.nanoTime();
		
		if(pn.converged == false){
			
			System.out.println("Training on node: " + pn.getID());
			
			// Get the data corresponding to this iteration
			INDArray cur_features = pn.trainSet.getFeatures();
			int num_batches = (int)cur_features.size(0)/pn.batch_size;
			int start_index = iter%num_batches;
			int end_index = start_index+1;
			
			
			System.out.println("Start Index: "+ start_index*pn.batch_size + " End Index: "+ end_index*pn.batch_size);
			System.out.println("NUM BATCHES: " + num_batches);
			
			////////////////////////////////////////////////////////////////////////////////////////////
			// Compute the features and labels for the nodes corresponding to the current batch
			INDArray features_pn = pn.trainSet.getFeatures().get(NDArrayIndex.interval(start_index*pn.batch_size,end_index*pn.batch_size), NDArrayIndex.all());
			INDArray labels_pn = pn.trainSet.getLabels().get(NDArrayIndex.interval(start_index*pn.batch_size,end_index*pn.batch_size), NDArrayIndex.all());
			
			
			
			pn.model.setInput(features_pn);
			
			
			// Determine activations of the last layer after going through the feedforward phase
			List<INDArray> activations_pn = pn.model.feedForward(true, false);
			
			INDArray predictions_pn = activations_pn.get(activations_pn.size() - 1);
			
			
			// Compute squared loss on both nodes, and then calculate the average loss vector.
			INDArray squared_loss_pn = crossEntropyLoss(predictions_pn, labels_pn);
			
			double score = squared_loss_pn.sumNumber().doubleValue();
			score /= squared_loss_pn.size(0);
			System.out.println("LOSS: "+score);
			
			double train_loss = pn.model.output(pn.trainSet.getFeatures()).sumNumber().doubleValue()/pn.trainSet.getFeatures().size(0);
			double test_loss = pn.model.output(pn.testSet.getFeatures()).sumNumber().doubleValue()/pn.testSet.getFeatures().size(0);
			System.out.println("Test loss: " + test_loss);
			System.out.println("Train loss: " + train_loss);
			int iteration = 0;
	        int epoch = 0;
	        pn.model.setInput(features_pn);
			Pair<Gradient, INDArray> p = pn.model.backpropGradient(squared_loss_pn, null);  //Calculate backprop gradient based on error array
			Gradient gradient_pn = p.getFirst();
			//Update the gradient: apply learning rate, momentum, etc
	        //This modifies the Gradient object in-place
	        pn.model.getUpdater().update(pn.model, gradient_pn, iteration, epoch, pn.batch_size, LayerWorkspaceMgr.noWorkspaces());
	        //Get a row vector gradient array, and apply it to the parameters to update the model
	        INDArray update_vector_pn = gradient_pn.gradient();
	        pn.model.params().subi(update_vector_pn);
	        
			
			
			////////////////////////////////////////////////////////////////////////////////////////////
			
	        long endTime = System.nanoTime();
	        pn.trainTime += (double)(endTime - startTime)/1e-9;
	        
	        
	        String csv_filename = resourcepath + "/run" + pn.numRun + "/vpnn_results.csv";
			System.out.println("Storing in " + csv_filename);
			String opString = pn.getID() + "," + iter + "," + train_loss +"," + test_loss +"," +pn.trainTime+"," + pn.readTime;
			
			// Write to file
			try {
			BufferedWriter bw = new BufferedWriter(new FileWriter(csv_filename, true));
			
			bw.write(opString);
			bw.write("\n");
			bw.close();
			}
			catch(Exception e) {
				
			}
	        
	        /*
			if (iter == 0) {
				pn.prevLoss = trainLoss;
				pn.convergedCount += 1;
			}
			
			else {
				if (trainLoss - pn.prevLoss <= pn.epsilon) {
					pn.convergedCount += 1;
				}
				if (pn.convergedCount == 1000000000) {
					pn.converged = true;
				}
			}
			*/	
	//long trainTimePerIter = System.nanoTime() - startTime;
	//pn.trainTime += trainTimePerIter;
		

		}
		
		
		//double trainTimeInDouble = (double)pn.trainTime/1e9;
		//double readInitTimeInDouble = (double)pn.readInitTime/1e9;
		
		
		
		
		
		
	}

	private List<Double> computeLoss(PegasosNode pn, String dataset_type) {
		/**
		 * Computes the loss and accuracy on a dataset.
		 */
		DataSet dataset = null;
		int start_index = 0;
		int end_index = 0;
		if (dataset_type == "train") {
			dataset = pn.trainSet;
			start_index = 0;
			end_index = pn.trainSet.numExamples();
		}
		else {
			dataset = pn.testSet;
			start_index = 0;
			end_index = pn.testSet.numExamples();
					
		}

		INDArray features = dataset.getFeatures().get(NDArrayIndex.interval(start_index*pn.batch_size,end_index*pn.batch_size), NDArrayIndex.all());
		INDArray labels = dataset.getLabels().get(NDArrayIndex.interval(start_index*pn.batch_size,end_index*pn.batch_size), NDArrayIndex.all());
		pn.model.setInput(features);
		List<INDArray> activations = pn.model.feedForward(false, false);
		
		INDArray predictions = activations.get(activations.size() - 1);
		INDArray loss_vector = SquaredLoss(predictions, labels);
		double score = loss_vector.sumNumber().doubleValue();
		score /= loss_vector.size(0);
		
		// Compute accuracy
		double acc = 0;
		for(int i=0; i<labels.length(); i++) {
			double gt = labels.getDouble(i);
			double pred = predictions.getDouble(i);
			if(Math.round(gt) == Math.round(pred)) {
				acc++;
			}
		}
		double accuracy = acc/labels.length();
		List<Double> result = new ArrayList<Double>();
		result.add(score);
		result.add(accuracy);
        return result;
		
	}
	
	private double computeAccuracy(PegasosNode pn, String dataset_type) {
		/**
		 * Computes the accuracy on a dataset.
		 */
		DataSet dataset = null;
		int start_index = 0;
		int end_index = 0;
		if (dataset_type == "train") {
			dataset = pn.trainSet;
			start_index = 0;
			end_index = pn.trainSet.numExamples();
		}
		else {
			dataset = pn.testSet;
			start_index = 0;
			end_index = pn.testSet.numExamples();
					
		}
		INDArray features = dataset.getFeatures().get(NDArrayIndex.interval(start_index*pn.batch_size,end_index*pn.batch_size), NDArrayIndex.all());
		INDArray labels = dataset.getLabels().get(NDArrayIndex.interval(start_index*pn.batch_size,end_index*pn.batch_size), NDArrayIndex.all());
		List<INDArray> activations = pn.model.feedForward(features);
		INDArray predictions = activations.get(activations.size() - 1);
		
		//TODO: Write a clause for argmax for multiclass
		int correctCount = 0;
		for(int i=0;i<=labels.length(); i++) {
			int label = 0;
			if (labels.getInt(i) >= 0.5){
				label = 1;
			}
			int pred = 0;
			if (predictions.getInt(i) >= 0.5){
				pred = 1;
			}
			
			System.out.println(label+ " " + pred);
			 if (label == pred) {
				 correctCount += 1;		 
			 }
			 
		}
		double accuracy = correctCount*100/labels.length();
		return accuracy;
		
	}
	
	

	/**
	 * Selects a random neighbor from those stored in the {@link Linkable} protocol
	 * used by this protocol.
	 */
	protected Node selectNeighbor(Node node, int pid) {
		Linkable linkable = (Linkable) node.getProtocol(lid);
		if (linkable.degree() > 0) 
			return linkable.getNeighbor(
					CommonState.r.nextInt(linkable.degree()));
		else
			return null;
	}

	public static void writeIntoFile(String millis) {
		File file = new File("exec-time.txt");
		 
		// if file doesnt exists, then create it
		if (!file.exists()) {
			try {
				file.createNewFile();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

		FileWriter fw;
		try {
			fw = new FileWriter(file.getAbsoluteFile(),true);

		BufferedWriter bw = new BufferedWriter(fw);
		bw.write(millis+"\n");
		bw.close();
		} catch (IOException e)
		
		 {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		

	}
	

}

